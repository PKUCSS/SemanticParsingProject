import argparse
from utils import *
from models import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_sdp_filename",type=str,default="dm.sdp")
    parser.add_argument("--test_file_name",type=str,default="test_data/esl.input")
    parser.add_argument("--output_path",type=str,default="test_results.sdp")
    args = parser.parse_args()
    make_final_submission(args.train_sdp_filename,args.test_file_name,args.output_path)
    print("submission results stored in {}".format(args.output_path))



    

