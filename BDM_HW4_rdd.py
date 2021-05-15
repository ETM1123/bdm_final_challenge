#  libraries 
from pyspark import SparkContext
from datetime import timedelta, datetime
import csv
import functools
import json
import numpy as np
import sys


# Part D
def filterPOIs(_, lines):
    ## lines is an genrator object - multiple lines
    reader = csv.reader(lines)
    # get contetns
    for line in reader:
        placekey, naics_code = line[0], line[9]
        if naics_code in CAT_CODES:
            yield placekey, CAT_GROUP[naics_code]




# Part F,G 
def extractVisits(storeGroup, _, lines):
    reader = csv.reader(lines)
    for line in reader:
        placekey, date_range_start, raw_visit_counts, visits_by_day = (line[0], 
                                                                 line[12],
                                                                 line[14],
                                                                 line[16])
        year = date_range_start[:4]
        if year in ['2019', '2020']:
            try:
                group = storeGroup[placekey]
                current_date = datetime.fromisoformat(date_range_start[:10])
                day_count = zip([((current_date + timedelta(days=i)) - datetime(2019,1,1)).days for i in range(7)], json.loads(visits_by_day))
                for day, count in day_count:
                    yield (group, day), count
            except:
                pass

# Part H,i
# Remember to use groupCount to know how long the visits list should be
def computeStats(groupCount, _, records):
    # records is an iterator -- > containg key, value pairs 
    for key, value in records:
        group, day = key[0], key[1]
        count = groupCount[group]
        current_count = len(value)
        diff = count - current_count 

        # extend values 
        values = list(value) + [0 for i in range(diff)]

        # get stats
        median = np.median(values)
        std = np.std(values)
        low, high = max(0, median - std), median + std

        # get date
        current_date = datetime(2019, 1, 1) + timedelta(days=day)
        if current_date.year == 2020:
            yield group, ','.join([str(current_date.year), current_date.strftime('%Y-%m-%d'), str(median), str(int(low)), str(int(high))])
        else:
            # project date to 2020
            yield group, ','.join([str(current_date.year), current_date.replace(year=2020).strftime('%Y-%m-%d'), str(median), str(int(low)), str(int(high))])



def main(sc):
    
    # Static variables
    CAT_CODES = set(['445210', '445110', '722410', '452311', '722513', 
                 '445120', '446110', '445299', '722515', '311811', 
                 '722511', '445230', '446191', '445291', '445220', 
                 '452210', '445292'])

    CAT_GROUP = {'452210': 0, '452311': 0, '445120': 1, '722410': 2, 
             '722511': 3, '722513': 4, '446110': 5, '446191': 5, 
             '722515': 6, '311811': 6, '445210': 7, '445299': 7, 
             '445230': 7, '445291': 7, '445220': 7, '445292': 7, 
             '445110': 8}

    OUTPUT_PREFIX = sys.argv[1]

    FILE_NAME = ['big_box_grocers',
           'convenience_stores',
           'drinking_places',
           'full_service_restaurants',
           'limited_service_restaurants',
           'pharmacies_and_drug_stores',
           'snack_and_bakeries',
           'specialty_food_stores',
           'supermarkets_except_convenience_stores']
  
    # RDD
    rddPlaces = sc.textFile('/data/share/bdm/core-places-nyc.csv')
    rddPattern = sc.textFile('/data/share/bdm/weekly-patterns-nyc-2019-2020/*')
  

    # Extract places of interst 
    rddD = rddPlaces.mapPartitionsWithIndex(filterPOIs).cache()

    # Compute the number of stores per group
    storeGroup = dict(rddD.collect())
    groupCount = rddD.map(lambda x: (x[1], 1)) \
      .reduceByKey(lambda x,y: x+y) \
      .sortByKey() \
      .map(lambda x: x[1]) \
      .collect()
    #groupCount

    # Filter the Pattter data and expload the vist_by_day list 
    rddG = rddPattern \
    .mapPartitionsWithIndex(functools.partial(extractVisits, storeGroup))

    # Compute daily stats for each group and convert data into csv format
    rddH = rddG.groupByKey() \
        .mapPartitionsWithIndex(functools.partial(computeStats, groupCount))

    # Sort data for output
    rddJ = rddH.sortBy(lambda x: x[1][:15])
    header = sc.parallelize([(-1, 'year,date,median,low,high')]).coalesce(1)
    rddJ = (header + rddJ).coalesce(10).cache()

    # Write data output for all groups
    for group in range(len(FILE_NAME)):
        rddJ.filter(lambda x: x[0] == group or x[0]==-1).values() \
        .saveAsTextFile(f'{OUTPUT_PREFIX}/{FILE_NAME[group]}')
        
if __name__=='__main__':
    sc = SparkContext()
    main(sc)

