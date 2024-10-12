import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from matplotlib.font_manager import FontProperties


def convert_minutes_to_hhmm(minutes):
    hours = minutes // 60
    remaining_minutes = minutes % 60
    return f"{hours:02}:{remaining_minutes:02}"

def plot_timeline(allocation, open_time, close_time):
    font_path = '/Library/Fonts/MyriadPro-Regular.otf'
    font_prop = FontProperties(fname=font_path)

    plt.rcParams['font.family'] = 'Montserrat-Italic'
    sns.set(style="whitegrid")

    fig, ax = plt.subplots(figsize=(12, 8))

    y_labels = []

    for i, a in enumerate(allocation):
        start_time = a[1]
        end_time = a[2]
        y_labels.append(f"Patient {a[0]}")

        ax.add_patch(
            patches.Rectangle(
                (start_time, i - 0.15),
                end_time - start_time,
                0.3
            )
        )

    ax.set_yticks(range(len(allocation)))
    ax.set_yticklabels(y_labels, fontsize=12, fontproperties=font_prop)
    ax.set_ylim(-0.5, len(allocation) - 0.5 + 0.5)
    ax.set_xlim([open_time, close_time])
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=10))
    ax.xaxis.set_minor_locator(plt.MaxNLocator(integer=True, nbins=20))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: convert_minutes_to_hhmm(int(x))))

    plt.xticks(fontsize=10, fontproperties=font_prop)
    plt.xlabel('Time', fontsize=14, fontproperties=font_prop)
    plt.ylabel('Patients', fontsize=14, fontproperties=font_prop)
    plt.title('Patient Infusion Schedule', fontsize=16, fontweight='bold', fontproperties=font_prop)
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    sns.despine(left=True, bottom=True)

    plt.tight_layout()
    plt.legend(loc='lower right', fontsize=10, prop=font_prop)
    plt.show()


def calculate_utilization(allocation, open_time, close_time, interval=10):
    time_range = range(open_time, close_time + 1, interval)
    utilization = []

    for time in time_range:
        count = sum(
            1 for a in allocation if a[1] <= time < a[2])
        utilization.append(count)

    return time_range, utilization


def calculate_orig_schedule_utilization(patients, open_time, close_time, interval=10):
    time_range = range(open_time, close_time + 1, interval)
    utilization = []

    for time in time_range:
        count = sum(
            1 for p in patients if p['readyTime'] <= time < p['readyTime'] + p['length'])
        utilization.append(count)

    return time_range, utilization

def plot_utilization(allocation, open_time, close_time, max_chairs):
    sns.set(style="whitegrid")
    font_path = '/Library/Fonts/MyriadPro-Regular.otf'
    font_prop = FontProperties(fname=font_path)

    plt.rcParams['font.family'] = 'Montserrat-Italic'

    time_range, utilization = calculate_utilization(allocation, open_time, close_time)

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(time_range, utilization, label='Utilization')
    ax.axhline(y=max_chairs, color='red', linestyle='--', label=f'Max Chairs ({max_chairs})')

    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=10))
    ax.xaxis.set_minor_locator(plt.MaxNLocator(integer=True, nbins=20))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: convert_minutes_to_hhmm(int(x))))

    plt.xticks(fontsize=10, fontproperties=font_prop)
    plt.yticks(fontsize=10, fontproperties=font_prop)
    plt.xlabel('Time', fontsize=14, fontproperties=font_prop)
    plt.ylabel('Number of Patients', fontsize=14, fontproperties=font_prop)
    plt.title('Infusion Center Utilization', fontsize=16, fontweight='bold', fontproperties=font_prop)
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    plt.legend(loc='lower right', fontsize=10, prop=font_prop)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.show()


def plot_nurse_timelines(allocation, num_nurses, open_time, close_time):
    font_path = '/Library/Fonts/MyriadPro-Regular.otf'
    font_prop = FontProperties(fname=font_path)

    plt.rcParams['font.family'] = 'Montserrat-Italic'
    sns.set(style="whitegrid")

    for nurse_id in range(num_nurses):
        fig, ax = plt.subplots(figsize=(12, 4))

        nurse_allocations = [alloc for alloc in allocation if alloc[4] == nurse_id]
        y_labels = []

        for i, (patient_id, start_time, end_time, chair_id, _) in enumerate(nurse_allocations):
            y_labels.append(f"Patient {patient_id} (Chair {chair_id})")
            ax.add_patch(
                patches.Rectangle(
                    (start_time, i - 0.15),
                    end_time - start_time,
                    0.3
                )
            )

        ax.set_yticks(range(len(nurse_allocations)))
        ax.set_yticklabels(y_labels, fontsize=12, fontproperties=font_prop)
        ax.set_ylim(-0.5, len(nurse_allocations) - 0.5 + 0.5)
        ax.set_xlim([open_time, close_time])
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=10))
        ax.xaxis.set_minor_locator(plt.MaxNLocator(integer=True, nbins=20))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: convert_minutes_to_hhmm(int(x))))

        plt.xticks(fontsize=10, fontproperties=font_prop)
        plt.xlabel('Time', fontsize=14, fontproperties=font_prop)
        plt.ylabel('Patients', fontsize=14, fontproperties=font_prop)
        plt.title(f'Nurse {nurse_id + 1} Assignment', fontsize=16, fontweight='bold', fontproperties=font_prop)
        plt.grid(axis='x', linestyle='--', alpha=0.7)

        sns.despine(left=True, bottom=True)

        plt.tight_layout()
        plt.legend(loc='lower right', fontsize=10, prop=font_prop)
        plt.show()


def plot_chair_timelines(allocation, num_stations, open_time, close_time):
    # Replace this with the path to your Century Gothic font file
    font_path = '/Library/Fonts/MyriadPro-Regular.otf'

    # Create a font properties object
    font_prop = FontProperties(fname=font_path)

    plt.rcParams['font.family'] = 'Montserrat-Italic'
    sns.set(style="whitegrid")

    fig, ax = plt.subplots(figsize=(12, 8))

    y_labels = [f"Chair {chair_id + 1}" for chair_id in range(num_stations)]

    for chair_id in range(num_stations):
        chair_allocations = [alloc for alloc in allocation if alloc[3] == chair_id]
        for (patient_id, start_time, end_time, _, nurse_id) in chair_allocations:
            ax.add_patch(
                patches.Rectangle(
                    (start_time, chair_id - 0.4),
                    end_time - start_time,
                    0.8,
                    edgecolor='black',
                    linewidth=0.5
                )
            )
            ax.text(
                (start_time + end_time) / 2, chair_id,
                f"{patient_id}",
                ha='center', va='center',
                fontsize=10, color='white', fontproperties=font_prop
            )

    ax.set_yticks(range(num_stations))
    ax.set_yticklabels(y_labels, fontsize=12, fontproperties=font_prop)
    ax.set_ylim(-0.5, num_stations - 0.5 + 0.5)
    ax.set_xlim([open_time, close_time])
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=10))
    ax.xaxis.set_minor_locator(plt.MaxNLocator(integer=True, nbins=20))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: convert_minutes_to_hhmm(int(x))))

    plt.xticks(fontsize=10, fontproperties=font_prop)
    plt.xlabel('Time', fontsize=14, fontproperties=font_prop)
    plt.ylabel('Chairs', fontsize=14, fontproperties=font_prop)
    plt.title('Chair Timelines', fontsize=16, fontweight='bold', fontproperties=font_prop)
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    plt.legend(loc='lower right', fontsize=10, prop=font_prop)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.show()