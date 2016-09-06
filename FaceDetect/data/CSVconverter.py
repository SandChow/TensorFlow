import os, argparse

def convert(file):
	with(open(file, 'r+')) as f:
		lines = f.readlines()
		f.seek(0)
		f.truncate()
		for line in lines:
			f.write(line.replace(" : ", ", "))

if __name__=="__main__":
	parser = argparse.ArgumentParser(description="Process a filepath.")
	parser.add_argument("filepath", help="a filepath of file to be converted into csv format.")
	args = parser.parse_args()
	convert(args.filepath)
	os.system("mv " + args.filepath + " " + args.filepath.split(".")[0] + ".csv")