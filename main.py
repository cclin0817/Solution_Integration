
✅ 還原內容如下：

#!/CAD/TCD_BE_CENTRAL/cclin/python/bin/python3
import tkinter as tk
from lib.application import Application
import argparse
import os
import logging
from lib import log

def parse_arguments():
    parser = argparse.ArgumentParser(description="TVC Application")
    parser.add_argument('-i', '--input_dir', type=str, required=False, help='Directory containing input qor files')
    parser.add_argument('-j', '--json_file', type=str, required=True, help='File path of merged JSON file')
    parser.add_argument('-o', '--output_dir', type=str, required=True, help='File path of output file')
    parser.add_argument('-c', '--checker', type=str, required=False, help='Checkers to run, enabled: distance, coverage, overlap, by default: skip')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(args.output_dir)

    # log setting
    log_file_name = 'Solution_Integration_GUI.log'
    log.set_log_config(args.output_dir+'/'+log_file_name)
    log.logger.info(f"Create {args.output_dir}/{log_file_name}")

    root = tk.Tk()
    root.title("TVC")
    app = Application(master=root, input_json=args.json_file, args=args, output_dir=args.output_dir)
    app.mainloop()

