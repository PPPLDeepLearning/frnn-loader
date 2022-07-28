import numpy as np


def find_rational(
    q,
    number,
    max_n=15,
    saving=False,
    save_path="/tigress/FRNN/signal_data_efit/d3d/EFITRT1/rational/",
):
    q = np.array(q)
    res_n = np.zeros(q.shape)
    res_m = np.zeros(q.shape)
    max_m = min(int(np.amax(q) * max_n), 70)
    for i in range(0, q.shape[0]):
        #   print(i)
        if i > 0 and np.mean(np.abs(q[i, :] - q[i - 1, :])) < 0.01:
            res_n[i, :] = res_n[i - 1, :]
            res_m[i, :] = res_m[i - 1, :]
            continue
        for j in range(q.shape[1] - 1):
            flag_found = False
            for n in np.arange(1, max_n):
                for m in np.arange(1, max_m):
                    qmn = m * 1.0 / n
                    if qmn > q[i, j + 1]:
                        break
                    elif q[i, j] < qmn:
                        if res_n[i, j] == 0:
                            res_n[i, j] = n
                            res_m[i, j] = m
                            break
                            flag_found = True
                # Possible bug: flag_found above is never executed since it comes after the breaks
                if flag_found == True:
                    # only record lowest order rational surface
                    break

    if saving == True:
        np.save(save_path + str(number), {"q": q, "n_mode": res_n, "m_mode": res_m})
    return res_n, res_m


def get_rational(
    q,
    number,
    max_n=15,
    saving=True,
    save_path="/tigress/FRNN/signal_data_efit/d3d/EFITRT1/rational/",
):
    try:
        dic = np.load(f"{save_path}{number}.npy", allow_pickle=True).item()
        res_n = dic["n_mode"]
        res_m = dic["m_mode"]
        return res_n, res_m
    except:
        print("calculating rational.....")
        n, m = find_rational(q, number, max_n=max_n, save_path=save_path, saving=saving)
        return n, m
