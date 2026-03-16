Contents of the README
1. How to use
2. Preprocessing approach
3. Data exploration


How to use
1. Create virtual environment (python3 -m venv venv)
2. Activate virtual environment (python3 venv/bin/activate) *note that this changes slightly if you are not using a mac
3. Install requirements.txt (pip3 install -r requirements.txt)
4. Run files (eg. python3 preprocess.py)


Preprocessing approach (all in preprocess.py)

1. Take raw.csv (the untouched data), and create a new file (processing.csv). All data is in the Data folder. Add a column on the right in processing.csv that is titled IN, and contains boolean values. In each row, the data will be used if it is marked as True in that column. All rows start with IN = 1.

2. Chunk the est_diameter_min (third column)into 500 equal ranges from the minimum value to the maximum value and plot those values on the X axis, with how many are in each range, on the y axis.  Find the median and the standard deviation, and highlight the regions on this graph that are 2+ standard deviations away from the mean. These points get marked as OUT (IN = 0) in processed.csv, and the plot is saved in the Plots folder.

3. Repeat step 2 for est_diameter_max, relative_velocity, miss_distance, and absolute_magnitude (columns 4-6, and 8 respectively). For each of these variables, all of the points are included in each plot, including the ones that are marked as OUT. This is to prevent shrinking the data more than outliers require.

4. Next this creates a new file (processed.csv) with the same data that was in processing.csv columns 1 (ID), 2 (name), 7(orbiting_body), and 8 (sentry_object) from the dataset (based on one indexed columns). This is because the IDs and the names are unnecessary and do not help us predict whether or not an asteroid is hazardous, and all of the objects have earth as their orbiting body and are not included in the sentry monitoring system. Therefore, those columns do not provide relevant predicitve information. In this file, remove all rows that were marked as OUT in processing.csv, and remove the IN column, since all of the remaining rows will have the same value.



Data exploration
1. Exploration.py is using pandas .corr() function to create a correlation matrix with every variable in processed.csv. This matrix is saved in Exploration.

2. Step 2 is creating another post_exploratory_data.csv in the Data folder. This is another post processing step that is modifying processed.csv. The correlation matrix shows that the extimated diameter min and the estimated diameter max are perfectly correlated, so post_exploratory_data will replace estimated_diameter_min and estimated_diameter_max with the average of the two for each data point, and the new column will simply be called estimated_diameter. Finally, the miss_distance only has 0.03 correlation with the outcome we are trying to predict. Since this correlation is insufficient for having predictive value, we are excluding this column in our post_exploratory_data.

3. Finally, in normalized.csv we normalize the remaining data in post_exploratory_data.csv by column (mean = 0, std = 1; this includes the hazardous column in the file, but unlike the other columns, we do not normalize this one). In std_means.csv, there are the original means and stds for each column of post_exploratory_data.csv (excluding the hazard column), so it can be unnormalized as desired.

4. The three remaining predictive columns are estimated_diameter, relative_velocity, and absolute magnitude. In step 4, we take normalized.csv, and we create 3 plots in our Exploration folder with each of those three predictors plotted against eachother in the plots. In the plots, points that have hazardous result True, are red, and points that have hazardous result False, are blue.