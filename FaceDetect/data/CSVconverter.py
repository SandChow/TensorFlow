import os, argparse

def convert(file):
	with(open(file)) as f:
		for line in f:
			line = line.replace(" : ", ", ")

if __name__=="__main__":
	parser = argparse.ArgumentParser(description="Process a filepath.")
	parser.add_argument("filepath", nargs=1,type="string", help="a filepath of file to be converted into csv format.")
	args = parser.parse_args()
	convert(args.filepath)
