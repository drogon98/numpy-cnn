import numpy as np

if __name__ == "__main__":
    arr_3d = np.array([
        [
            [101,102,103,104,105],
            [201,202,203,204,205],
            [301,302,303,304,305],
        ],
        [
            [401,402,403,404,405],
            [501,502,503,504,505],
            [601,602,603,604,605],
        ],
        [
            [701,702,703,704,705],
            [801,802,803,804,805],
            [901,902,903,904,905],
        ]
    ])
    # print(arr_3d)
    # print(arr_3d.shape)
    exact_sliced = arr_3d[:]
    # print(exact_sliced)
    a= arr_3d[:,:,2]
    print(a)
    print(a.shape)