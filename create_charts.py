from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


OUTPUT_PATH = Path("output").resolve()
RUNS = Path("20240630110454")


OUTPUT_PATH.glob

PERF1_NAME = "ILP"
PERF2_NAME = "Approximation"
PERF3_NAME = "Standard"


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
    return df


def main(folder = OUTPUT_PATH / RUNS):
    # Loop through all benchmarks in 
    for benchmark in get_benchmarks(folder):
        print("Starting to process benchmark: ", benchmark.stem)
        # How man time and cost files are there for this benchmark?
        n_files = len(list(get_time_cost_pair(benchmark)))

        # Create a figure and axis for the benchmark which contain n_files subplots (as a grid with at most 2 columns)
        fig, axs = plt.subplots(n_files // 2 + n_files % 2, 2, figsize=(15, 5 * (n_files // 2 + n_files % 2)))
        # Flatten the axs array
        axs = axs.flatten()

        # Add a title to the figure
        fig.suptitle(f"Benchmark: {benchmark.stem}", fontsize=16)
        
        # Loop through all time and cost files for this benchmark
        i = 0
        for timefile, costfile in get_time_cost_pair(benchmark):
            print("Processing Pair: ", timefile.stem, costfile.stem)
            # Process the time and cost files
            df = df_from_timecost(timefile, costfile)
            # Compute performance stats
            df = compute_performance_stats(df)
            # Plot performance stats for the three algorithms as a cdf for the respective perf columns
            
            for col in ["perf1", "perf2", "perf3"]:
                sns.ecdfplot(data=df[col], ax=axs[i])
            axs[i].set_title("ECDF of Performance Factor")
            axs[i].legend([PERF1_NAME, PERF2_NAME, PERF3_NAME])
            # Add a grid to the plot
            axs[i].grid()
            # Add a tiltle to the subplot
            axs[i].set_title("Configuration: " + timefile.stem.split("_")[-1])
            # Set the x-axis label
            axs[i].set_xlabel("Performance Factor")
            # Set the y-axis label
            axs[i].set_ylabel("CDF")



            # Increment the index
            i += 1

        # Cut the figure so that there is no empty space surrounding the plots
        plt.tight_layout()


        # Save the figure
        plt.savefig(str(OUTPUT_PATH / RUNS) + f"/{benchmark.stem}.png", dpi=300)

        plt.show()

          
if __name__ == "__main__":
    main()