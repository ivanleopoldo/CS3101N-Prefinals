import numpy as np


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
        dct = {key: val if val != "m" else None for key, val in zip(attr, vals)}
        obj_list.append(dct)

    return obj_list


def interpolation(obj_list) -> list:
    for i in range(1, len(obj_list) - 1):
        for key, value in obj_list[i].items():
            if (
                value is None
                and obj_list[i - 1][key] is not None
                and obj_list[i + 1][key] is not None
            ):
                obj_list[i][key] = (
                    float(obj_list[i - 1][key]) + float(obj_list[i + 1][key])
                ) / 2

    return obj_list


def convert_to_matrix(obj_list) -> list:
    return [[row[attr] for row in obj_list] for attr in list(obj_list[0].keys())[2:]]


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
        col = [float(row[i]) for row in matrix if row[i] is not None]

        if len(set(col)) == 1:
            continue

        mean = calculate_mean(col)
        std_dev = calculate_std_dev(mean, col)

        matrix = [
            [(float(cell) - mean) / std_dev if cell is not None else 0 for cell in row]
            for row in matrix
        ]
    return matrix


def calculate_covariance(row, col) -> float:
    size = len(row)
    mean_row = calculate_mean(row)
    mean_col = calculate_mean(col)
    return sum((row[i] - mean_row) * (col[i] - mean_col) for i in range(size)) / (
        size - 1
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


def transpose_matrix(matrix) -> list:
    return [[j[i] for j in matrix] for i in range(len(matrix[0]))]


def dot_product(a, b) -> list:
    return sum(float(x) * float(y) for x, y in zip(a, b))


def matrix_multiplication(a, b):
    return [[dot_product(row, col) for col in b] for row in a]


def power_iteration(matrix, n=1000):
    n_rows = len(matrix)
    n_cols = len(matrix[0])
    vector = np.random.rand(n_cols)

    for _ in range(n):
        vector = np.dot(matrix, vector)
        magnitude = np.linalg.norm(vector)
        vector /= magnitude

    return magnitude, vector


def pca(matrix, n):
    covar_matrix = covariance(matrix)
    n_features = len(matrix[0])

    eigenvalues, eigenvectors = [], []

    for _ in range(n):
        eigenvalue, eigenvector = power_iteration(covar_matrix)
        eigenvalues.append(eigenvalue)
        eigenvectors.append(eigenvector)

        deflated_matrix = [
            [eigenvalue * v1 * v2 for v1, v2 in zip(eigenvectors[-1], eigenvectors[-1])]
            for _ in range(len(covar_matrix))
        ]
        covar_matrix = [
            [
                covar_matrix[i][j] - deflated_matrix[i][j]
                for j in range(len(covar_matrix[0]))
            ]
            for i in range(len(covar_matrix))
        ]

    return matrix_multiplication(matrix, transpose_matrix(eigenvectors[:n]))


def svd(matrix):
    matrix = [
        [float(val) if val != None else float(0) for val in row] for row in matrix
    ]

    return np.linalg.svd(matrix, full_matrices=False)


if __name__ == "__main__":
    attr, data = read_file("./V4 data/2020.arff")
    parsedData = parse_data(attr, data)
    interpolatedData = interpolation(parsedData)

    matrix = convert_to_matrix(interpolatedData)

    standardizedData = standardize(matrix)
    n_components = 2
    result = pca(standardizedData, n_components)
    for row in result[0:5]:
        print(row)

    for row in svd(standardizedData):
        print(row)


# from sklearn.decomposition import PCA
# import pandas as pd
# import numpy as np

# matrix = [[float(val) if val != None else float(0) for val in row] for row in matrix]
# df = pd.DataFrame(np.matrix([[float(cell) for cell in j] for j in matrix]))
# df_std = (df - df.mean()) / (df.std())
# pca = PCA(n_components=2)
# pc = pca.fit_transform(df_std)
# pdf = pd.DataFrame(data=pc)
