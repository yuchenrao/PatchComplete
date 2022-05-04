import csv
import sys

def map_label_to_cadid(filename):
    """
    This function maps ScanNet labels to ShapeNet cadids

    :param filename: csv file for mapping ScanNet labels to ShapeNet cadids
    :type filename: str
    :return label_to_cadid: a dict which maps ScanNet labels to ShapeNet cadids
    :rtype: dict
    """
    label_to_cadid = {}
    with open(filename) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter='\t')
        for row in csv_reader:
            catid = row['wnsynsetid']
            if len(catid) != 0:
                label_to_cadid[row['raw_category'].replace(' ', '-')] = catid[1:]
    return label_to_cadid

def print_error(message, user_fault=False):
    """
    This function prints an error message and quit

    :param message: error message
    :type message: str
    """
    sys.stderr.write('ERROR: ' + str(message) + '\n')
    # if user_fault:
    #   sys.exit(2)
    # sys.exit(-1)
