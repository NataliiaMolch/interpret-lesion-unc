import os
import logging
import pandas as pd


def save_options(args, filepath):
    with open(filepath, 'w') as f:
        for arg, value in sorted(vars(args).items()):
            f.write("%s:\t%r\n" % (arg, value))


class Logger:
    def __init__(self, log_file=None):
        if log_file is None:
            self.log_file = None
            logging.info("Logs will appear in the console and won't be saved")
        else:
            self.log_file = log_file
            self.df = pd.DataFrame([])
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            logging.info(f"Logs will appear in the {log_file}.")

    def write(self, msg):
        if self.log_file is None: print(msg)
        else:
            if isinstance(msg, dict):
                self.df = pd.concat([self.df, pd.DataFrame([msg])], axis=0)
                self.df.to_csv(self.log_file)
            else:
                logging.warn(f"Unable to concat the msg, output to console: {msg}")
