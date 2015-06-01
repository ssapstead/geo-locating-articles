#!/bin/bash

cd ../data

grep -vE ",U" labelled_data.csv > labelled_data_no_U.csv

cp labelled_data_no_U.csv converted_data_labelled.csv

sed -i 's/T/1/g' converted_data_labelled.csv
sed -i 's/F/0/g' converted_data_labelled.csv
sed -i 's/,Y/,1/g' converted_data_labelled.csv
sed -i 's/,N/,0/g' converted_data_labelled.csv


exit
