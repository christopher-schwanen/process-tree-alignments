from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np  

OUTPUT_PATH = Path("output").resolve()
RUNS = Path("final_sepsis")


OUTPUT_PATH.glob

PERF1_NAME = "MILP (Gurobi)"
PERF2_NAME = "Approximation"
PERF3_NAME = "Standard"
PLOT_STANDARD_ARTIFICIAL = False

AXISFONTSIZE = 11
FONTSIZE = 16
LINEWIDTH = 1.5
DPI = 600

TIMEOUT = 20
INFINITY = float('inf')
INFINITY = 10 ** 10


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
    # We need to exclude the cases where the time is greater than the timeout
    df["perf1"] = df["perf1"].where(df["time1"] <= TIMEOUT, INFINITY)
    df["perf2"] = df["perf2"].where(df["time2"] <= TIMEOUT, INFINITY)
    df["perf3"] = df["perf3"].where(df["time3"] <= TIMEOUT, INFINITY)
    return df


def main(folder = OUTPUT_PATH / RUNS):
    # Loop through all benchmarks in 
    for benchmark in get_benchmarks(folder):
        print("Starting to process benchmark: ", benchmark.stem)
        # How man time and cost files are there for this benchmark?
        n_files = len(list(get_time_cost_pair(benchmark)))

        # Plot title:
        plot_title = f"Benchmark: {benchmark.stem}"
        plot_figsize=(15, 5)
        plot_xlabel = "Performance Factor"
        plot_ylabel = "Cumulative Probability"
        plot_axtitle = "ECDF of Performance Factor"



        # if there are more than one time/cost pairs, draw them in a grid
        if n_files > 1:

            
            # Create a figure and axis for the benchmark which contain n_files subplots (as a grid with at most 2 columns)
            fig, axs = plt.subplots(n_files // 2 + n_files % 2, 2, figsize=(15, 5 * (n_files // 2 + n_files % 2)))
            # Flatten the axs array
            axs = axs.flatten()

            # Add a title to the figure
            fig.suptitle(plot_title, fontsize=FONTSIZE)
            
            # Loop through all time and cost files for this benchmark
            i = 0
            for timefile, costfile in get_time_cost_pair(benchmark):                
                
                print("Processing Pair: ", timefile.stem, costfile.stem)
                # Process the time and cost files
                df = df_from_timecost(timefile, costfile)
                # Compute performance stats
                df = compute_performance_stats(df)

                # Compute the range of performance factors expcept for INFINITY
                perf1 = df["perf1"].where(df["perf1"] != INFINITY).dropna()
                perf2 = df["perf2"].where(df["perf2"] != INFINITY).dropna()
                perf3 = df["perf3"].where(df["perf3"] != INFINITY).dropna()

                min_perf = min(perf1.min(), perf2.min(), perf3.min())
                max_perf = np.ceil(max(perf1.max(), perf2.max(), perf3.max()))

                # Plot performance stats for the three algorithms as a cdf for the respective perf columns
                
                rel_cols = ["perf1", "perf2", "perf3"]
                cols_names = [PERF1_NAME, PERF2_NAME, PERF3_NAME]

                for col in rel_cols:
                    sns.ecdfplot(data=df[col], ax=axs[i],  linewidth=LINEWIDTH)
                axs[i].set_title(plot_axtitle)
                axs[i].legend(cols_names)
                # Add a grid to the plot
                axs[i].grid()
                # Add a tiltle to the subplot
                axs[i].set_title("Configuration: " + timefile.stem.split("_")[-1])
                # Set the x-axis label
                axs[i].set_xlabel(plot_xlabel)
                # Set the y-axis label
                axs[i].set_ylabel(plot_ylabel)

                # Set the x-axis limits
                axs[i].set_xlim(0.9, max_perf)

                # Set the font size of the x-axis and y-axis labels
                axs[i].set_xlabel(plot_xlabel, fontsize=AXISFONTSIZE)
                axs[i].set_ylabel(plot_ylabel, fontsize=AXISFONTSIZE)


                # Increment the index
                i += 1

        else:
            # Case of a single time/cost pair
            # Create a figure and axis for the benchmark
            fig, ax = plt.subplots(1, 1, figsize=plot_figsize)

            # Add a title to the figure
            fig.suptitle(plot_title, fontsize=FONTSIZE)

            # Process the time and cost files
            for timefile, costfile in get_time_cost_pair(benchmark):
                print("Processing Pair: ", timefile.stem, costfile.stem)
                # Process the time and cost files
                df = df_from_timecost(timefile, costfile)
                # Compute performance stats
                df = compute_performance_stats(df)


                # Plot Standard?
                if PLOT_STANDARD_ARTIFICIAL:
                    rel_cols = ["perf1", "perf2", "perf3"]
                    cols_names = [PERF1_NAME, PERF2_NAME, PERF3_NAME]
                else:
                    rel_cols = ["perf1", "perf2"]
                    cols_names = [PERF1_NAME, PERF2_NAME]

                # Plot performance stats for the three algorithms as a cdf for the respective perf columns
                for col in rel_cols:
                    sns.ecdfplot(data=df[col], ax=ax, linewidth=LINEWIDTH)
                ax.set_title(plot_axtitle)
                ax.legend(cols_names)
                # Add a grid to the plot
                ax.grid()
                # Add a tiltle to the subplot
                ax.set_title("Configuration: " + timefile.stem.split("_")[-1])
                # Set the x-axis label
                ax.set_xlabel(plot_xlabel)
                # Set the y-axis label
                ax.set_ylabel(plot_ylabel)

                



        # Cut the figure so that there is no empty space surrounding the plots
        plt.tight_layout()


        # Save the figure
        plt.savefig(str(OUTPUT_PATH / RUNS) + f"/{benchmark.stem}.png", dpi=DPI)

        plt.show()

          
if __name__ == "__main__":
    main()