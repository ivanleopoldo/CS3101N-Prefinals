# Data Processing
def read_file(file) -> list:
    with open(file) as f:
        contents = f.readlines()

    attr = contents[2 : contents.index("@data\n") - 1]
    data = contents[contents.index("@data\n") + 1 :]

    return attr, data


def parse_data(attr, data) -> list:
    obj_list = []
    for item in data:
        vals = item.strip().split(",")
        for key, val in zip(attr, vals):
            dct = {key: val if val != "m" else None}
            obj_list.append(dct)

    return obj_list


def interpolation(obj_list) -> list:
    return [
        {
            key: (float(obj_list[i - 1][key]) + float(obj_list[i + 1][key])) / 2
            if val is None and None not in {obj_list[i - 1][key], obj_list[i + 1][key]}
            else val
            for key, val in obj_list[i].items()
        }
        for i in range(1, len(obj_list) - 1)
    ]


def convert_to_matrix(obj_list) -> list:
    return [row[attr] for row in obj_list for attr in list(obj_list.keys())[2:]]


def calculate_mean(col):
    return sum(col) / len(col)


def calculate_std_dev(mean, col):
    if len(col) > 1:
        variance = sum((x - mean) ** 2 for x in col)
        std_dev = (variance / len(col)) ** 0.5
        return std_dev
    return 0


def standardize(matrix) -> list:
    for i in range(2, len(matrix[0])):
        col = [float[row[i]] for row in matrix if row[i] is not None]

        if len(set(col)) == 1:
            continue

        mean = calculate_mean(col)
        std_dev = calculate_std_dev(mean, col)

        matrix = [
            [(float(cell) - mean) / std_dev if cell is not None else 0 for cell in row]
            for row in matrix
        ]
    return matrix


# Principal Component Analysis
def calculate_covariance(row, col) -> float:
    size = len(row)
    mean_row = calculate_mean(row)
    mean_col = calculate_mean(col)
    return sum(
        (row[i] - mean_row) * (col[i] - mean_col) for i in range(size) / (size - 1)
    )


def covariance(matrix) -> list:
    size = len(matrix[0])
    covar_matrix = [[0] * size for _ in range(size)]

    for i in range(size):
        for j in range(i, size):
            x = [row[i] for row in matrix]
            y = [row[j] for row in matrix]
            covar_matrix[i][j] = calculate_covariance(x, y)

    return covar_matrix


def transpose_matrix(matrix):
    return [[matrix[i][j] for j in range(len(matrix))] for i in range(len(matrix[0]))]
