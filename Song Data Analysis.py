
### Import all the necessary modules for data analysis
import warnings
warnings.simplefilter("ignore")

# Clears all variable values previously set
from IPython import get_ipython
get_ipython().magic('reset -sf')

# Provides ways to work with large multidimensional arrays
import numpy as np 
# Allows for further data manipulation and analysis
import pandas as pd
import seaborn as sns
# In Anaconda -> Environments -> Not Installed -> pandas-datareader -> Apply
from pandas_datareader import data as web # Reads stock data 
import matplotlib.pyplot as plt # Plotting
import matplotlib.dates as mdates # Styling dates
%matplotlib inline

### Import Songs csv file for analysis
csv_file = 'dbin_songs_data_v2.csv'
try:
    dbin_songs_data_v2_df = pd.read_csv(csv_file, parse_dates=False)
except FileNotFoundError:
    pass
    print("File Does not Exist")
else:
    print(f"The {csv_file} File was read successfully")

dbin_songs_data_v2_df
# Confirming if the dataframe is empty
df=dbin_songs_data_v2_df
df.empty
dup = df.duplicated().sum()
dup
# Checking the first 10 records of the dataframe
df.head(10)
# Checking how many records and columns we imported
df.shape
# Listing all the column headers of our dataframe
df.columns
# Use the info() function to check the data structure of your dataframe

df.info()

# [int8: -128 to 127]; [unint8: 0 to 255]; [int16: -32,768 to 32,767]; [uint16: 0 to 65,537]
# [int32: -2,147,483,648 to-2,147,483,647]; [unint32: 0 to 4,294,967,296]; 
# [int64: -9,223,372,036,854,776 to -9,223,372,036,854,775]; [uint64: 0 to 18,446,744,073,709,551,616]
# [float8, float16, float32, float32, float64] follows similar pattern but no unsigned float [unfloat]
df.describe()
# Converting all the columns into the appropriate datatype
string_columns = ['title', 'artist', 'album']
for column in string_columns:
    dbin_songs_data_v2_df[column] = dbin_songs_data_v2_df[column].astype ("string")

string_columns = ['song_id', 'year_released']
for column in string_columns:
    dbin_songs_data_v2_df[column] = dbin_songs_data_v2_df[column].astype ("int16")

string_columns = ['duration', 'tempo', 'loudness']
for column in string_columns:
    dbin_songs_data_v2_df[column] = dbin_songs_data_v2_df[column].astype ("float16")
    df = dbin_songs_data_v2_df
# Use the info() function to check the data structure of your dataframe

df.info()
# Checking the number of null values in each column
df.isnull().sum()
df.duplicated().sum()
# Songs with year of release equal to zero are erroneous, using pandas inbuilt function, 
# how many songs have  incorrect year of release?

# Counting the number of songs with a year of release equal to zero
incorrect_year_count = df[df['year_released'] == 0].shape[0]

print(f"Number of songs with incorrect year of release: {incorrect_year_count}")
# The loudness of a song must be less than zero. The closer the value is to zero, 
# the louder the song. How many songs have loudness greater than  or equal to zero?

# Counting the number of songs with loudness greater than or equal to zero
incorrect_loudness_count = df[df['loudness'] >= 0].shape[0]

print(f"Number of songs with loudness greater than or equal to zero: {incorrect_loudness_count}")

# The tempo of a song cannot be zero, How many songs have a tempo of zero?

# Counting the number of songs with a tempo of zero
incorrect_tempo_count = df[df['tempo'] == 0].shape[0]

print(f"Number of songs with a tempo of zero: {incorrect_tempo_count}")

# Using the panda's drop function, delete all the songs with year = 0

# Identify rows with year_released equal to zero
rows_to_drop = df[df['year_released'] == 0].index

# Drop these rows from the DataFrame
df.drop(rows_to_drop, inplace = True) 

# Print the first few rows to verify the changes
print(df.head())


# Confirm if rows with year released is empty 
rows_to_drop = df[df['year_released'] == 0]
rows_to_drop.empty
# Using the panda's drop function, delete all the songs with tempo = 0

# Identify rows with tempo equal to zero
rows_to_drop = df[df['tempo'] == 0].index

# Drop these rows from the DataFrame
df.drop(rows_to_drop, inplace = True)

# Print the first few rows to verify the changes
print(df.head())


# Using the panda's drop function, delete all the songs with loudness >= 0

# Identify rows with loudness greater than or equal to zero
rows_to_drop = df[df['loudness'] >= 0].index

# Drop these rows from the DataFrame
df.drop(rows_to_drop, inplace = True)

# Print the first few rows to verify the changes
print(df.head())

df.shape
# Create a pivot table with year_released as the index, loudness as the values, and mean as the aggregation function
pivot_table = df.pivot_table(index='year_released', values='loudness', aggfunc='mean')

# Display the pivot table
print(pivot_table)


# Using matplotlib, draw a line graph of the year against the average loudness

import matplotlib.pyplot as plt

# Create a pivot table with year_released as the index, loudness as the values, and mean as the aggregation function
pivot_table = df.pivot_table(index='year_released', values='loudness', aggfunc='mean')

# Plot the pivot table using Matplotlib
plt.figure(figsize=(10, 6))
plt.plot(pivot_table.index, pivot_table['loudness'], marker='o')
plt.title('Average Loudness by Year')
plt.xlabel('Year Released')
plt.ylabel('Average Loudness')
plt.grid(True)
plt.show()
df
#Plotting top seven artist by count of songs
song_art = df.groupby('artist')['song_id'].count().sort_values(ascending = False).reset_index()
song_art = song_art.head(7)
print(song_art) 
#the graph
plt.figure(figsize=(10, 4))
sns.barplot(x = 'artist', y='song_id', data = song_art)
plt.title('Top 7 Artist by Count of Songs')
plt.xlabel('Artists')
plt.ylabel('Number of Songs')
plt.tick_params(axis='x', labelrotation=45)
plt.grid(True)
plt.show()
df.describe()
#Boxplot of song loudness
plt.figure(figsize=(10, 4))
sns.boxplot(df['loudness'], orient ='h')
plt.title('Boxplot of Songs Loudness')
plt.xlabel('Loudness')
#plt.ylabel('Number of Songs')
plt.tick_params(axis='y', labelrotation=45)
plt.grid(True)
plt.show()
#Distribution of Song DUration
plt.figure(figsize=(10, 4))
sns.distplot(df['duration'], bins = 100)
sns.kdeplot(df['duration'], color = 'red')
plt.title('Distribution and Density Estimate of Song Duration')
plt.xlabel('Duration')
plt.grid(which='major', axis='y')
plt.show()
# range of years in the dataset
print(f"Max Years - {max(df['year_released'])},\
	Min Years - {min(df['year_released'])}")

# Grouping the songs by the decades
range = [1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, np.inf]
labels = ['1920s', '1930s','1940s',
          '1950s','1960s','1970s',
          '1980s','1990s', '2000s']

remapping_data = df.copy()
remapping_data['Decades'] = pd.cut(remapping_data['year_released'],
										bins=range,
										labels=labels)

remapping_data.head()
#Song records per year
plt.figure(figsize=(10, 6))
remapping_data['Decades'].value_counts().plot.bar()
plt.title('Song Records per Decade')
plt.xlabel('Decade Released')
plt.ylabel('Number of Songs')
plt.grid(True)
plt.show()
# Save the graph as a picture in your folder

# Create a pivot table with year_released as the index, loudness as the values, and mean as the aggregation function
pivot_table = df.pivot_table(index='year_released', values='loudness', aggfunc='mean')

# Plot the pivot table using Matplotlib
plt.figure(figsize=(10, 6))
plt.plot(pivot_table.index, pivot_table['loudness'], marker='o')
plt.title('Average Loudness by Year')
plt.xlabel('Year Released')
plt.ylabel('Average Loudness')
plt.grid(True)

# Save the plot as an image file
output_image_path = 'average_loudness_by_year.png'
plt.savefig(output_image_path)

# Display the plot
plt.show()
**The graph above shows that the loudness was at its peak (-5) on the year 1920 also between 1920 and 1940 
we observed a significant drop in the loudness (-30)
Then, Loudness has been increasing over the time. 
This may be due to advances in technology and sound engineering over the years.**

#pip install openpyxl
#pip install xlsxwriter
import xlsxwriter
# Save the clean song datafram as an excel file
#import xmltodict
file_name  = 'dbin_songs_data_v2_df_cleaned.xlsx'
df.to_excel(file_name, index = False)

# crtl + / -- Used to comment out codes.

# file_path = r'C:\Users\User\Desktop\Data Training\input files\input_files\dbin_songs_data_cleaned.xlsx'
# df.to_excel(file_path, index = False)
