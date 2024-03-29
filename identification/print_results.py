import os
import csv
import argparse

def parse_arguments(parser):
    parser.add_argument("-d", "--dataset_path", default="", required=True)
    args = parser.parse_args()
    return args

def print_results(args):
        results_location = os.path.join(args.dataset_path, "output.csv")

        with open(results_location, 'r') as infile:
            results_csv = csv.DictReader(infile)
            output_data = 0.0
            correct_id = 0.0
            for row in results_csv:
                output_data += 1.0
                if row['predicted-fnsku'] == row['gt-fnsku']:
                     correct_id += 1.0
                
            id_retrieval_rate = correct_id/output_data * 100
            print ('id_retrieval_rate: ', id_retrieval_rate)
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)

    print_results(args)