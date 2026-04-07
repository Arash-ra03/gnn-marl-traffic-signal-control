import numpy as np


def generate_periods(mu, var, duration=3600, num_intervals=3600):
    """
    Generate a list of periods for SUMO's randomTrips.py --period option.

    Parameters:
    - mu: Mean of the normal distribution for arrival rates.
    - var: Variance of the normal distribution for arrival rates.
    - duration: Total time duration to generate periods for (default: 3600 seconds).
    - num_intervals: Number of subintervals to divide the duration into (default: 3600).

    Returns:
    - A list of periods (in seconds) suitable for use with --period option.
    """

    # Generate normal distribution arrival rates
    arrival_rates = np.random.normal(loc=mu, scale=np.sqrt(var), size=num_intervals)

    # Convert arrival rates to periods
    periods = np.round(1.0 / np.clip(arrival_rates, a_min=0.0001, a_max=None), 3)  # more the rounding more corruption of normal dist!

    # Ensure the periods are within a reasonable range
    periods = np.clip(periods, a_min=0.1, a_max=10.0)  # Modify these limits as needed

    return periods.tolist()


def format_periods(periods):
    """
    Format the periods for the --period option.

    Parameters:
    - periods: List of periods to format.

    Returns:
    - A string formatted for the --period option.
    """
    return ','.join(map(str, periods))


# Example usage
mu = 2.75  # Mean arrival rate per second
var = 1  # Variance of the arrival rate
duration = 9000  # Total duration in seconds
num_intervals = 9000  # Number of intervals (typically one per second)

periods = generate_periods(mu, var, duration, num_intervals)
formatted_periods = format_periods(periods)
with open(r'.\normal_flows_275_1_9000.txt', 'w') as f:
    f.write(formatted_periods)
# print(f"--period=\"{formatted_periods}\"")
