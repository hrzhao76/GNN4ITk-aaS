import argparse
import subprocess
from pathlib import Path

def run_perf_analyzer(model_name, input_data_path, percentile, protocol, initial_measurement_interval, max_attempts):
    output_csv_name = Path(f'perf_output.csv')
    nth_attempt = 0
    while not output_csv_name.exists() and nth_attempt < max_attempts:
        command = [
            'perf_analyzer',
            '-m', model_name,
            '--percentile', str(percentile),
            '-i', protocol,
            '--input-data', input_data_path,
            '--measurement-interval', str(initial_measurement_interval),
            '-f', str(output_csv_name),  
            '--verbose-csv',
            '--collect-metrics',
        ]

        try:
            result = subprocess.run(command, text=True, capture_output=True, check=True)
            print(f"Attempt {nth_attempt + 1}: Success")
            break
        except subprocess.CalledProcessError as e:
            print(f"Attempt {nth_attempt + 1}: Failed, doubling measurement interval")
            initial_measurement_interval *= 2 

    print("Test completed.")

def main():
    parser = argparse.ArgumentParser(description='Run perf_analyzer with configurable parameters.')
    parser.add_argument('--model_name', type=str, default='GNN4ITk_MM_Infer', help='Model name for perf_analyzer.')
    parser.add_argument('--input_data_path', type=str, default='/global/cfs/cdirs/m3443/data/GNN4ITk-aaS/dev_mm/data/testset/testset.json', help='Path to the input data file.')
    parser.add_argument('--percentile', type=int, default=95, help='Percentile for latency measurement.')
    parser.add_argument('--protocol', type=str, default='grpc', help='Communication protocol to use.')
    parser.add_argument('--initial_measurement_interval', type=int, default=100000, help='Initial measurement interval in microseconds.')
    parser.add_argument('--max_attempts', type=int, default=5, help='Maximum number of attempts to run perf_analyzer.')
    
    args = parser.parse_args()
    
    run_perf_analyzer(args.model_name, args.input_data_path, args.percentile, args.protocol, args.initial_measurement_interval, args.max_attempts)

if __name__ == "__main__":
    main()
