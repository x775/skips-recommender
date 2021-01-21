import csv

HEADER = 0
STATIC_FOLDER = "data/"


def read_file(filename):
    content = {}
    with open(STATIC_FOLDER + filename, "r") as source:
        for line in source:
            row = [l.strip() for l in line.split("\t")]
            content[row[0]] = tuple(row[1:])
    return content


def read_csv(filename):
    with open(STATIC_FOLDER + filename, "r", encoding="utf-8") as source:
        reader = csv.reader(source)
        return list(reader)


def read_listening_events(filename):
    content = []
    with open(STATIC_FOLDER + filename, "r") as source:
        for line in source:
            row = line.split(",")
            content.append((row[1], row[2], row[3]))
    return set(content)


def read_tags(filename):
    content = {}
    with open(STATIC_FOLDER + filename, "r") as source:
        for line in source:
            row = [e.strip() for e in line.split(",")]
            content[row[0]] = row[1]


def write_list_to_disk(data, filename):
    with open(filename, "a") as outfile:
        outfile.write("\n".join(data))


def write_to_disk(data, filename, mode="a"):
    with open(filename, mode) as outfile:
        if type(data) is dict:
            writer = csv.writer(outfile)
            for row in data.items():
                writer.writerow(row)
        elif type(data) is list: 
            writer = csv.DictWriter(outfile, list(data[0].keys()))
            global HEADER
            if not HEADER:
                writer.writeheader()
                HEADER = 1
            for row in data:
                writer.writerow(row)