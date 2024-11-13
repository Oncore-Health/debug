import itertools
import random
from datetime import datetime, timedelta
import random
import numpy as np
import math
import pulp
from pymongo import MongoClient

def connect_to_mongo(uri):
    client = MongoClient(uri, tlsAllowInvalidCertificates=True)
    return client

uri = "mongodb+srv://test:gytc2elpQxENixcU@testcluster.jzglo.mongodb.net/?retryWrites=true&w=majority&appName=TestCluster"
excel_name = 'IC_Scheduling_Guidelines_And_Start_Times_6-6-2024.xlsx'

db_name = 'test-dev'
# Connect to MongoDB and update the schedule
client = connect_to_mongo(uri)
db = client[db_name]

settings_collection = db['settings']

# Fetch data from MongoDB once
settings_data = settings_collection.find_one()

# Maintain a set of patient names used on the current day
used_patient_names = set()


def reset_patient_names():
    global used_patient_names
    used_patient_names = set()


def generate_unique_random_name():
    while True:
        name = generate_random_name()
        if name not in used_patient_names:
            used_patient_names.add(name)
            return name


def convert_minutes_to_hhmm(minutes):
    hours = minutes // 60
    remaining_minutes = minutes % 60
    return f"{hours:02}:{remaining_minutes:02}"


def generate_example_data():
    # Define the number of chairs, max patients per nurse, and operating hours

    scale = 1.3

    num_chairs = int(20 * scale)
    M = 4  # Example nursing capacity (max number of patients a nurse can monitor at once)
    num_patients = int(65 * scale)
    num_nurses = int(10 * scale)
    open_time = 480
    close_time = 1080
    break_start_time = 600  # Breaks can start at 10:00 (600 minutes from midnight)
    break_end_time = 840  # Breaks can end at 2:00 (840 minutes from midnight)
    break_duration = 30  # Break duration in minutes

    # Generate patient list
    patients = []
    appointment_durations = {
        '1_hour': int(20 * scale),  # 20 patients with 1-hour appointments
        '2_hours': int(15 * scale),  # 15 patients with 2-hour appointments
        '3-5_hours': int(10 * scale),  # 10 patients with 3-5 hour appointments
        '6-8_hours': int(4 * scale),  # 4 patients with 6-8 hour appointments
        '9+_hours': int(1 * scale)  # 1 patient with 9+ hour appointments
    }

    appointment_times = {
        '1_hour': 60,
        '2_hours': 120,
        '3-5_hours': random.choice([180, 240, 300]),
        '6-8_hours': random.choice([360, 420, 480]),
        '9+_hours': random.choice([540, 600])
    }

    patient_id = 0
    for duration_type, count in appointment_durations.items():
        for _ in range(count):
            duration = appointment_times[duration_type]
            max_start_time = close_time - duration

            # Ensure start time is in 10-minute intervals
            start_time = random.randint(open_time // 10, max_start_time // 10) * 10

            acuity = random.randint(1, 3)
            patients.append({
                'patientId': patient_id,
                'readyTime': start_time,
                'length': duration,
                'dueTime': close_time,
                'acuity': acuity
            })
            patient_id += 1

    # Generate nursing schedule
    nurses = []
    center_open_hours = (close_time - open_time) // 60

    for i in range(num_nurses):
        shift_length = random.randint(8, center_open_hours)  # Shift length between 8 hours and the remaining open hours
        latest_start_hour = center_open_hours - shift_length
        start_hour = random.randint(0, latest_start_hour)
        shift_start = open_time + start_hour * 60
        shift_end = shift_start + shift_length * 60

        nurses.append({
            'nurseId': i,
            'startTime': shift_start,
            'endTime': shift_end
        })

    print(f"Number of Nurses: {num_nurses}")
    print(f"Number of Chairs: {num_chairs}")
    print(f"Maximum Patient Capacity per Nurse: {M}")
    print(f"Operating Hours: {convert_minutes_to_hhmm(open_time)} - {convert_minutes_to_hhmm(close_time)}")
    print("Patients:")
    for patient in patients:
        print(
            f"Patient ID: {patient['patientId']}, Ready Time: {convert_minutes_to_hhmm(patient['readyTime'])}, Due Time: {convert_minutes_to_hhmm(patient['dueTime'])}, Duration: {patient['length']} minutes, Acuity: {patient['acuity']}")

    print("Nurses:")
    for nurse in nurses:
        print(
            f"Nurse ID: {nurse['nurseId']}, Shift Start: {convert_minutes_to_hhmm(nurse['startTime'])}, Shift End: {convert_minutes_to_hhmm(nurse['endTime'])}")

    return num_nurses, num_chairs, M, open_time, close_time, patients, nurses, break_start_time, break_end_time, break_duration

# Returns index of x in arr if present, else -1
def binary_search(arr, low, high, x):
    # Check base case
    if high >= low:
        mid = (high + low) // 2

        # If element is present at the middle itself
        if arr[mid] == x:
            return mid
        # If element is smaller than mid, then it can only
        # be present in left subarray
        elif arr[mid] > x:
            return binary_search(arr, low, mid - 1, x)
        # Else the element can only be present in right subarray
        else:
            return binary_search(arr, mid + 1, high, x)
    else:
        # Element is not present in the array
        return -1


def find_treatment(acuity_table, treatment_name):
    # Extract the list of treatment names for binary search
    treatment_names = [treatment['name'].lower() for treatment in acuity_table]

    # Find the index using recursive binary search
    index = binary_search(treatment_names, 0, len(treatment_names) - 1, treatment_name.lower())
    # If the index is valid, return the treatment
    if index != -1:
        # Find the actual treatment using the name
        return acuity_table[index]
    else:
        return None  # Treatment not found


def get_random_time_slot(break_start_time, break_end_time, break_duration):
    # Generate a list of valid start times at 10-minute intervals within the given range
    valid_start_times = [time for time in range(break_start_time, break_end_time, 10)
                         if time + break_duration <= break_end_time]

    if not valid_start_times:
        raise ValueError("No valid time slot available within the given range.")

    # Pick a random start time from the list of valid start times
    random_start_time = random.choice(valid_start_times)
    random_end_time = random_start_time + break_duration

    return random_start_time, random_end_time


def generate_booked_days_in_advance():
    random_value = random.random()

    if random_value < 0.05:
        # 5% of cases: on the day of the appointment
        booked_days_in_advance = 0
    elif random_value < 0.20:
        # 15% of cases: less than 2 weeks out (1 to 13 days)
        booked_days_in_advance = np.random.randint(1, 14)
    else:
        # 80% of cases: more than 2 weeks out (14+ days)
        booked_days_in_advance = np.random.randint(14, 43)

    return booked_days_in_advance


def generate_random_mrn():
    return ''.join(random.choices(string.digits, k=10))


def generate_random_name():
    first_names = [
        'John', 'Emma', 'James', 'Olivia', 'William', 'Ava', 'Noah', 'Sophia',
        'Liam', 'Isabella', 'Mason', 'Mia', 'Ethan', 'Amelia', 'Lucas', 'Charlotte',
        'Alexander', 'Harper', 'Henry', 'Evelyn', 'Michael', 'Abigail', 'Benjamin',
        'Emily', 'Sebastian', 'Ella', 'Jackson', 'Elizabeth', 'Aiden', 'Sofia',
        'Matthew', 'Avery', 'Samuel', 'Chloe', 'David', 'Lily', 'Joseph', 'Scarlett',
        'Carter', 'Aria', 'Jacob', 'Madison', 'Logan', 'Grace', 'Oliver', 'Hannah',
        'Elijah', 'Zoe', 'Daniel', 'Luna', 'Matthew', 'Penelope', 'Owen', 'Layla',
        'Levi', 'Ellie', 'Luke', 'Nora', 'Isaac', 'Hazel', 'Gabriel', 'Riley',
        'Anthony', 'Lillian', 'Jayden', 'Violet', 'Dylan', 'Aurora', 'Jackson',
        'Maya', 'Caleb', 'Savannah', 'Isaiah', 'Scarlet', 'Nathan', 'Stella',
        'Sebastian', 'Vera', 'Miles', 'Lucy', 'Connor', 'Eleanor', 'Wyatt', 'Aria'
    ]

    last_names = [
        'Smith', 'Johnson', 'Williams', 'Jones', 'Brown', 'Davis', 'Miller', 'Wilson',
        'Moore', 'Taylor', 'Anderson', 'Thomas', 'Jackson', 'White', 'Harris',
        'Martin', 'Thompson', 'Garcia', 'Martinez', 'Robinson', 'Clark', 'Rodriguez',
        'Lewis', 'Lee', 'Walker', 'Hall', 'Allen', 'Young', 'King', 'Wright',
        'Scott', 'Torres', 'Nguyen', 'Hill', 'Flores', 'Green', 'Adams', 'Nelson',
        'Baker', 'Gonzalez', 'Hernandez', 'Mitchell', 'Perez', 'Roberts', 'Campbell',
        'Sanchez', 'Turner', 'Parker', 'Carter', 'Phillips', 'Evans', 'Edwards',
        'Collins', 'Stewart', 'Morris', 'Reed', 'Cook', 'Morgan', 'Bell', 'Murphy',
        'Bailey', 'Rivera', 'Cooper', 'Richardson', 'Cox', 'Howard', 'Ward',
        'Brooks', 'Sanders', 'Price', 'Bennett', 'Wood', 'Barnes', 'Ross', 'Henderson',
        'Coleman', 'Jenkins', 'Perry', 'Powell', 'Long', 'Patterson', 'Hughes'
    ]

    return f"{random.choice(first_names)} {random.choice(last_names)}"



def generate_realistic_data():
    acuity_table = settings_data['acuityTable']

    chairs = settings_data['chairs']
    open_time = settings_data['openTime']
    close_time = settings_data['closeTime']
    break_start_time = settings_data['break_start_time']
    break_end_time = settings_data['break_end_time']
    nurses_mongo = list(settings_data['nurses'].values())[:16]
    break_duration = settings_data['break_duration']
    types = ["add-on", "cancelled", "no-show", "N/A"]
    probabilities = [0.08, 0.08, 0.08, 0.76]

    # Convert time string to a datetime object
    open_time_obj = datetime.strptime(open_time['0'], "%H:%M")
    close_time_obj = datetime.strptime(close_time['0'], "%H:%M")
    break_start_time_obj = datetime.strptime(break_start_time, "%H:%M")
    break_end_time_obj = datetime.strptime(break_end_time, "%H:%M")

    # Calculate the number of minutes after midnight
    open_time = open_time_obj.hour * 60 + open_time_obj.minute
    close_time = close_time_obj.hour * 60 + close_time_obj.minute

    break_start_time = break_start_time_obj.hour * 60 + break_start_time_obj.minute
    break_end_time = break_end_time_obj.hour * 60 + break_end_time_obj.minute

    active_chairs = [chair for chair in chairs if chair['status']]

    # Define the number of chairs, max patients per nurse, and operating hours
    num_chairs = len(active_chairs)

    M = 4  # Example nursing capacity (max number of patients a nurse can monitor at once)

    num_nurses = len(nurses_mongo)

    # Generate patient list
    scale = 1.6
    patients = []
    appointment_durations = {
        '1_hour': int(20 * scale),  # 20 patients with 1-hour appointments
        '2_hours': int(15 * scale),  # 15 patients with 2-hour appointments
        '3-5_hours': int(10 * scale),  # 10 patients with 3-5 hour appointments
        '6-8_hours': int(4 * scale),  # 4 patients with 6-8 hour appointments
        '9+_hours': int(1 * scale)  # 1 patient with 9+ hour appointments
    }

    num_patients = sum(appointment_durations.values())

    appointment_times = {
        '1_hour': 60,
        '2_hours': 120,
        '3-5_hours': random.choice([180, 240, 300]),
        '6-8_hours': random.choice([360, 420, 480]),
        '9+_hours': random.choice([540, 600])
    }

    excel_data = pd.read_excel(excel_name)
    department_dict = excel_data.set_index('Medication/Procedure')['Department'].to_dict()

    department_dict = {k: str(v) for k, v in department_dict.items()}

    total_acuity = 0

    clinic_providers = ['Dr. Smith', 'Dr. Patel', 'Dr. Doe', 'Dr. Ross', 'Dr. Brown', 'Dr. Shaw', 'Dr. Jane']
    referring_departments = [
        'Hematology/Oncology (HemOnc)', 'Radiation Oncology', 'Surgical Oncology', 'Gynecologic Oncology',
        'Neuro-Oncology', 'Pediatric Oncology', 'Thoracic Oncology', 'Breast Oncology',
        'Gastrointestinal Oncology', 'Genitourinary Oncology'
    ]

    patient_id = 0
    for duration_type, count in appointment_durations.items():
        added = 0
        while added < count:
            duration = appointment_times[duration_type]
            max_start_time = close_time - duration

            # Ensure start time is in 10-minute intervals
            start_time = random.randint(open_time // 10, max_start_time // 10) * 10

            # Filter infusion types based on the duration
            matching_infusions = [infusion for infusion in acuity_table if infusion['duration'] == duration]
            if not matching_infusions:
                matching_infusions = [infusion for infusion in acuity_table if
                                      abs(infusion['duration'] - duration) <= 60]  # Tolerance of +/- 1 hour

            # Select a random infusion from the matching list
            selected_infusion = random.choice(matching_infusions)
            acuity = selected_infusion['value']

            # Generate 'bookedDate'
            booked_days_in_advance = generate_booked_days_in_advance()
            booked_date = datetime.now() - timedelta(days=booked_days_in_advance)
            booked_day = booked_date.strftime("%m/%d/%Y")

            department_string = None
            for key in department_dict:
                if selected_infusion['name'] in key:
                    department_string = department_dict[key]
                    break  # Stop once you find the first match

            if department_string:
                # Split the department string on '/' to get options
                department_options = department_string.split('/')

                # Randomly select one of the department options
                referring_department = random.choice(department_options)
            else:
                # print("No matching infusion found in the department dictionary.")
                # print(selected_infusion)
                continue

            patient_name = generate_unique_random_name()
            patient_mrn = generate_random_mrn()
            # Generate random 'type' once for consistency
            patient_type = np.random.choice(types, p=probabilities)
            referring_department_random = random.choice(referring_departments)
            clinic_provider = random.choice(clinic_providers)

            added += 1

            total_acuity += acuity

            patients.append({
                'patientId': patient_id,
                'patientName': patient_name,
                'patientMRN': patient_mrn,
                'readyTime': start_time,
                'length': duration,
                'dueTime': close_time,
                'acuity': acuity,
                'infusionType': selected_infusion['name'],
                'department': referring_department,
                'type': patient_type,
                'referringDepartment': referring_department_random,
                'clinicProvider': clinic_provider,
                'bookedDate': booked_day,
                'originalInfo': {'patientName': patient_name,
                                 'patientMRN': patient_mrn,
                                 'readyTime': start_time,
                                 'length': duration,
                                 'dueTime': close_time,
                                 'acuity': acuity,
                                 'infusionType': selected_infusion['name'],
                                 'department': referring_department,
                                 }
            })
            patient_id += 1

    # Generate nursing schedule
    nurses = []
    center_open_hours = (close_time - open_time) // 60

    nurse_data = {
        'google-oauth2|0': [450, 1200],
        'google-oauth2|1': [450, 1200],
        'google-oauth2|2': [450, 1200],
        'google-oauth2|3': [450, 1200],
        'auth0|6723f8381a769e4773c771d8': [450, 1200],
        'google-oauth2|5': [450, 1200],
        'google-oauth2|6': [450, 1200],
        'google-oauth2|7': [450, 1200],
        'google-oauth2|8': [450, 1200],
        'google-oauth2|9': [450, 1200],
        'google-oauth2|101311321465229661693': [450, 1200],
        'google-oauth2|112343536148519048621': [450, 1080],
        'google-oauth2|12': [450, 1080],
        'google-oauth2|13': [450, 1080],
        'google-oauth2|14': [450, 1080],
        'google-oauth2|15': [450, 960],
        # Add more entries as needed
    }

    for nurse in nurses_mongo:
        shift_length = random.randint(8, center_open_hours)
        latest_start_hour = center_open_hours - shift_length
        start_hour = random.randint(0, latest_start_hour)
        shift_start = open_time + start_hour * 60
        shift_end = shift_start + shift_length * 60
        lunch_break_start, lunch_break_end = get_random_time_slot(break_start_time, break_end_time, break_duration)
        print(nurse_data, nurse['nurseId'])
        nurses.append({
            'id': nurse['id'],
            'nurseId': nurse['nurseId'],  # Use the actual nurseId from MongoDB
            'nurseName': nurse['nurseName'],  # Use the nurse name from MongoDB
            'nurseEmail': nurse['nurseEmail'],  # Use the nurse email from MongoDB
            'startTime': nurse_data[nurse['nurseId']][0],
            'endTime': nurse_data[nurse['nurseId']][1],
            'lunchBreakStart': lunch_break_start,
            'lunchBreakEnd': lunch_break_end,
        })

    # print(f"Number of Patients: {num_patients}")
    # print(f"Number of Nurses: {num_nurses}")
    # print(f"Number of Chairs: {num_chairs}")
    # print(f"Maximum Patient Capacity per Nurse: {M}")
    # print(f"Operating Hours: {convert_minutes_to_hhmm(open_time)} - {convert_minutes_to_hhmm(close_time)}")
    # print("Patients:")
    # for patient in patients:
    #     print(
    #         f"Patient ID: {patient['patientId']}, Patient Name: {patient['patientName']}, Infusion Type: {patient['infusionType']}, Ready Time: {convert_minutes_to_hhmm(patient['readyTime'])}, Due Time: {convert_minutes_to_hhmm(patient['dueTime'])}, Duration: {patient['length']} minutes, Acuity: {patient['acuity']}, Referring Department: {patient['department']}")
    #
    # print("Nurses:")
    # for nurse in nurses:
    #     print(
    #         f"Nurse ID: {nurse['nurseId']}, Nurse Name: {nurse['nurseName']}, Nurse Email: {nurse['nurseEmail']}, Shift Start: {convert_minutes_to_hhmm(nurse['startTime'])}, Shift End: {convert_minutes_to_hhmm(nurse['endTime'])}")
    #
    # print("------------------------")

    return num_nurses, num_chairs, M, open_time, close_time, patients, nurses, break_start_time, break_end_time, break_duration, active_chairs


import pandas as pd
import random
import string


def generate_naive_allocation(patients, nurses, num_chairs, open_time, close_time):
    patients_sorted = sorted(patients, key=lambda p: p['readyTime'])
    allocation = []
    chair_end_times = [open_time] * num_chairs
    nurse_index = 0
    num_nurses = len(nurses)

    for patient in patients_sorted:
        patient_id = patient['patientId']
        ready_time = patient['readyTime']
        duration = patient['length']

        # Find the chair with the earliest available end time
        earliest_chair_id = chair_end_times.index(min(chair_end_times))
        start_time = max(ready_time, chair_end_times[earliest_chair_id])
        end_time = start_time + duration

        # Assign the patient to this chair
        chair_end_times[earliest_chair_id] = end_time

        # Assign the next nurse in a round-robin fashion
        nurse_id = nurses[nurse_index]['nurseId']
        nurse_index = (nurse_index + 1) % num_nurses

        allocation.append((patient_id, start_time, end_time, earliest_chair_id, nurse_id))

    return allocation
