import math
import pulp
import json
from pathlib import Path
import numpy as np
import random

# Directory containing the JSON files
data_dir = Path("generated_data")


def schedule_patients_no_set_lunch(patients, num_stations, num_nurses, open_time, close_time, M, nurses_mongo):
    # Define the problem
    prob = pulp.LpProblem("InfusionCenterSchedulingWithErrors", pulp.LpMaximize)

    time_slots = range(open_time, close_time, 10)

    # Variables: x[p,t] = 1 if appointment p starts at timeslot t
    x = pulp.LpVariable.dicts("StartTimeslot", [(p['patientId'], t) for p in patients for t in time_slots],
                              cat='Binary')

    # Objective: Minimize total weighted deferring time and makespan
    deferring_time_weight = 1
    makespan_weight = 1

    prob += (
        deferring_time_weight * pulp.lpSum(
            [((t - p['readyTime']) * x[p['patientId'], t]) for p in patients for t in time_slots]
        ) +
        makespan_weight * pulp.lpSum(
            [((t + p['length']) * x[p['patientId'], t]) for p in patients for t in time_slots]
        )
    )

    # Constraints

    for p in patients:
        prob += pulp.lpSum([x[p['patientId'], t] for t in time_slots]) == 0
        prob += pulp.lpSum([x[p['patientId'], t] for t in range(open_time, p['readyTime'], 10)]) == 0 

    for t in time_slots:
        prob += pulp.lpSum([x[p['patientId'], t] for p in patients for t_prime in
                            range(max(open_time, t - p['length']), t + 1, 10)]) <= num_stations

        prob += (
            pulp.lpSum([x[p['patientId'], t_prime] for p in patients for t_prime in
                        range(max(open_time, t - p['length']), t + 1, 10)]) * (1 / M) +
            pulp.lpSum([x[p['patientId'], t] for p in patients]) * (1 - (1 / M)) +
            pulp.lpSum([x[p['patientId'], t - p['length']] for p in patients if t - p['length'] in time_slots]) * (
                1 - (1 / M))
            <= num_nurses + 2
        )

    # Solve the problem
    prob.solve(pulp.PULP_CBC_CMD(timeLimit=60))

    if prob.status == pulp.LpStatusOptimal:

        # Extract the solution and assign patients to stations and nurses
        allocation = []
        chair_assignments = [0] * num_stations  # Tracks end time of the current assignment for each chair
        nurse_capacities = {nurse['nurseId']: 0 for nurse in
                            nurses_mongo}  # Tracks current capacity usage for each nurse
        nurse_end_times = {nurse['nurseId']: [] for nurse in
                           nurses_mongo}  # Tracks end times of patients assigned to each nurse

        # Initialize direction and nurse index variables
        forward_direction = True
        nurse_index = 3
        num_nurses = len(nurses_mongo)

        for t in time_slots:
            for p in sorted(patients, key=lambda k: k['patientId']):  # Keep the sorted order for patients
                if x[p['patientId'], t].varValue == 1:
                    # Assign to the first available chair
                    for chair_id in range(num_stations):
                        chair_assignments[chair_id] = t + p['length']
                        break  # Stop after assigning to the first chair, even if occupied

                    # Assign nurses in a zigzag (round-robin) fashion
                    if forward_direction:
                        nurse_index += 2  # Skip nurses every second time
                        if nurse_index >= num_nurses:  # Wrap around the nurse index
                            nurse_index = nurse_index % num_nurses
                        nurse = nurses_mongo[nurse_index]
                        nurse_id = nurse['nurseId']
                    else:
                        nurse_index -= 2  # Go backward, skipping nurses
                        if nurse_index < 0:  # Wrap around the nurse index
                            nurse_index = num_nurses - 1
                        nurse = nurses_mongo[nurse_index]
                        nurse_id = nurse['nurseId']

                    # Assign to this randomly chosen nurse, regardless of their capacity
                    nurse_capacities[nurse_id] += 1
                    nurse_end_times[nurse_id].append(t + p['length'])
                    allocation.append((p['patientId'], t, t + p['length'], chair_id, nurse_id))
            # Nurse capacities remain overloaded

        return allocation
    else:
        return None

def calculate_roi_metrics(allocation, patients, nurses):
    # hint : we have to use allocation as that is the scheduling done by the algorithm
    overtime_per_nurse = []
    patient_wait_times = []

    # right now we are just seeing how long the shift is of a nurse
    for n in nurses:
        scheduled_start = n['startTime']
        scheduled_end = n['endTime']

        total_overtime = max(0, scheduled_end - scheduled_start - 480)

        overtime_per_nurse.append(total_overtime)

    avg_overtime_per_nurse = np.mean(overtime_per_nurse) if overtime_per_nurse else 0

    # right now we are randomly giving wait time per patient and not calculating
    for p in patients:
        wait_time = random.randint(-30, 60) 
        patient_wait_times.append(wait_time)

    avg_patient_wait_time = np.mean(patient_wait_times) if patient_wait_times else 0

    metrics = {
        'avgOvertimePerNurse': avg_overtime_per_nurse,
        'avgPatientWaitTime': avg_patient_wait_time,
    }

    return metrics

def load_from_json(filename):
    """
    Load data from a JSON file.
    """
    filepath = data_dir / filename
    with open(filepath, "r") as file:
        data = json.load(file)
    return data

def main():
    """
    Main function to load data from JSON files and process it.
    """
    print("Loading data from JSON files...")

    # Load data from JSON files
    patients = load_from_json("patients.json")
    nurses = load_from_json("nurses.json")
    schedule_data = load_from_json("schedule.json")

    # Extract details from the schedule_data
    num_nurses = schedule_data["num_nurses"]
    num_chairs = schedule_data["num_chairs"]
    M = schedule_data["max_patients_per_nurse"]
    open_time = schedule_data["open_time"]
    close_time = schedule_data["close_time"]
    break_start_time = schedule_data["break_start_time"]
    break_end_time = schedule_data["break_end_time"]
    break_duration = schedule_data["break_duration"]
    active_chairs = schedule_data["active_chairs"]

    print("Data loaded successfully.")

    allocation = schedule_patients_no_set_lunch(patients, num_chairs, num_nurses, open_time, close_time, M, nurses)


    for alloc in allocation:
        print(f"Patient {alloc[0]} starts at {alloc[1]} and ends at {alloc[2]} in chair {alloc[3]} with nurse {alloc[4]}")

    print("ROI FROM OPTIMIZED SCHEDULE")
    optimized_roi = calculate_roi_metrics(
        allocation, patients, nurses)

    print(optimized_roi)


if __name__ == "__main__":
    main()
