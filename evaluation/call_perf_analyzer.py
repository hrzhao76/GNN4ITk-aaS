import argparse
import subprocess
from pathlib import Path
import logging


def run_perf_analyzer(
    model_name,
    input_data_path,
    percentile,
    protocol,
    use_measurement_interval,
    measurement_request_count,
    concurrency_range,
    max_attempts,
    output_path,
    output_csv_name,
):
    measurement_mode = "time_windows" if use_measurement_interval else "count_windows"
    logging.info(
        f"""Running perf_analyzer with the following parameters:
        model_name={model_name}, 
        input_data_path={input_data_path}, 
        percentile={percentile}, 
        protocol={protocol}, 
        use_measurement_interval={use_measurement_interval},
        measurement_mode={measurement_mode},
        measurement_request_count={measurement_request_count},
        concurrency_range={concurrency_range},
        max_attempts={max_attempts},
        output_path={output_path},
        output_csv_name={output_csv_name}"""
    )

    output_csv_path = Path(output_path) / output_csv_name
    nth_attempt = 0
    while not output_csv_path.exists() and nth_attempt < max_attempts:
        command = [
            "perf_analyzer",
            "-m",
            model_name,
            "--percentile",
            str(percentile),
            "--concurrency-range",
            concurrency_range,
            "-i",
            protocol,
            "--input-data",
            input_data_path,
            "--measurement-mode",
            measurement_mode,
            "--measurement-request-count",
            str(measurement_request_count),
            "-f",
            str(output_csv_path),
            "--verbose-csv",
            "--collect-metrics",
        ]

        logging.info(f"Start running perf_analyzer, attempt {nth_attempt + 1}")
        try:
            nth_attempt += 1
            result = subprocess.run(command, text=True, capture_output=True, check=True)
            logging.info(f"Attempt {nth_attempt}: Success")
            break
        except subprocess.CalledProcessError as e:
            logging.error(f"Attempt {nth_attempt}: Failed, error: {e.stderr}")
            if use_measurement_interval:
                measurement_request_count *= 2

    if not output_csv_name.exists():
        logging.error("Test failed after maximum attempts.")
    else:
        logging.info("Test completed successfully.")


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Run perf_analyzer with configurable parameters."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="GNN4ITk_MM_Infer",
        help="Model name for perf_analyzer.",
    )
    parser.add_argument(
        "--input_data_path",
        type=str,
        default="/global/cfs/cdirs/m3443/data/GNN4ITk-aaS/dev_mm/data/testset/testset.json",
        help="Path to the input data file.",
    )
    parser.add_argument(
        "--percentile", type=int, default=95, help="Percentile for latency measurement."
    )
    parser.add_argument(
        "--protocol", type=str, default="grpc", help="Communication protocol to use."
    )
    parser.add_argument(
        "--use_measurement_interval",
        action="store_true",
        help="Use measurement interval or count for perf_analyzer.",
    )
    parser.add_argument(
        "--measurement_request_count",
        type=int,
        default=10,
        help="Number of requests to send for each measurement, default is 10.",
    )
    parser.add_argument(
        "--concurrency_range",
        type=str,
        default="1:1:1",
        help="Concurrency range for perf_analyzer, [start:end:step]. Example: 1:1:1",
    )
    parser.add_argument(
        "--max_attempts",
        type=int,
        default=5,
        help="Maximum number of attempts to run perf_analyzer.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=".",
        help="Path to save the output CSV file.",
    )
    parser.add_argument(
        "--output_csv_name",
        type=str,
        default="perf_output.csv",
        help="Name of the output CSV file.",
    )

    args = parser.parse_args()

    run_perf_analyzer(
        args.model_name,
        args.input_data_path,
        args.percentile,
        args.protocol,
        args.use_measurement_interval,
        args.measurement_request_count,
        args.concurrency_range,
        args.max_attempts,
        args.output_path,
        args.output_csv_name,
    )


if __name__ == "__main__":
    main()
