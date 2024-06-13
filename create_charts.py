from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


OUTPUT_PATH = Path("output").resolve()
RUNS = Path("20240613183823")


OUTPUT_PATH.glob

def get_benchmarks(folder: Path):
    # Get the benchmarks
    for directory in folder.iterdir():
        if directory.is_dir():
            yield directory

def get_time_cost_pair(folder: Path):
    # Get the time and cost files   
    for timefile in folder.glob("times*.csv"):
        for costfile in folder.glob(timefile.stem.replace("times", "costs") + ".csv"):
                yield timefile, costfile

def df_from_timecost(timefile: Path, costfile: Path):
    # Read the time and cost files
    time_df = pd.read_csv(timefile, header=None)
    time_df.columns = columns=["time1", "time2", "time3", "trace"]
    cost_df = pd.read_csv(costfile, header=None)
    cost_df.columns = columns=["cost1", "cost2", "cost3", "trace"]
    # Now merge the two dataframes on column "trace"
    df = pd.merge(time_df, cost_df, on="trace")
    print(df.columns)
    # Simple preprocessing: group by trace and take minimum of each column
    df = df.groupby("trace").min()
    # Round cost 1 column to 2 decimal places
    df["cost1"] = df["cost1"].round(2)
    return df


def compute_performance_stats(df: pd.DataFrame):
    # For each row of df, take the minimum of the time columns and divide each time value in this row by this minimum 
    # Store results in new columns perf1 perf2 perf3
    df["perf1"] = df["time1"] / df[["time1", "time2", "time3"]].min(axis=1)
    df["perf2"] = df["time2"] / df[["time1", "time2", "time3"]].min(axis=1)
    df["perf3"] = df["time3"] / df[["time1", "time2", "time3"]].min(axis=1)
    print(df.head(2))
    return df


def main(folder = OUTPUT_PATH / RUNS):
    # Loop through all benchmarks in 
    for benchmark in get_benchmarks(folder):
        print("Starting to process benchmark: ", benchmark.stem)
        # Loop through all time and cost files
        for timefile, costfile in get_time_cost_pair(benchmark):
            print("Processing Pair: ", timefile.stem, costfile.stem)
            # Process the time and cost files
            df = df_from_timecost(timefile, costfile)
            # Compute performance stats
            df = compute_performance_stats(df)
            # Plot performance stats for the three algorithms as a cdf for the respective perf columns
            
            fig, ax = plt.subplots(figsize=(10, 5))
            for col in ["perf1", "perf2", "perf3"]:
                sns.ecdfplot(data=df[col], ax=ax)
            ax.set_title("ECDF of Performance")
            ax.legend(["perf1", "perf2", "perf3"])
            plt.show()

          
if __name__ == "__main__":
    main()