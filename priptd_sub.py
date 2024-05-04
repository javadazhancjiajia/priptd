import math
import os
from phe import paillier

import numpy as np


class Center:
    def __init__(self):
        pub_k, pri_k = paillier.generate_paillier_keypair()
        self.pub_k = pub_k
        self.pri_k = pri_k

    def get_pub_key(self):
        return self.pub_k

    def get_pri_key(self):
        return self.pri_k


class Server:
    def __init__(self, pri_k):
        self.pri_k = pri_k
        self.data = None
        self.m = 0

    def upload(self, ciphertexts: list):
        plaintexts = []
        for ele in ciphertexts:
            plaintexts.append(self.pri_k.decrypt(ele))
        return plaintexts

    def upload_plaintext(self, plaintexts: list):
        if self.data is None:
            self.data = plaintexts
        else:
            for i in range(plaintexts.__len__()):
                self.data[i] += plaintexts[i]

    def cal(self, ciphertexts: list):
        rst = 0
        for ciphertext in ciphertexts:
            rst += self.pri_k.decrypt(ciphertext)
        return rst

    def truth_discovery(self, flag: int, data: list, weights: list = None, adjs: list = None):
        truth_set = []
        if flag == 1:
            for j in range(data[0].__len__()):
                s = 0
                for i in range(data.__len__()):
                    s += data[i][j]
                truth_set.append(s / data.__len__())
            return truth_set
        elif flag == 2:
            sum_weight = sum(weights)

            for j in range(data[0].__len__()):
                sum_truth = 0
                for i in range(data.__len__()):
                    sum_truth = data[i][j] * weights[i] + sum_truth
                truth_set.append(sum_truth / sum_weight)
            return truth_set
        elif flag == 3:
            sum_weight = sum(weights)

            for j in range(data[0].__len__()):
                sum_truth = 0
                for i in range(data.__len__()):
                    sum_truth = data[i][j] * (weights[i] + adjs[i]) + sum_truth
                truth_set.append(sum_truth / sum_weight)
            return truth_set
        else:
            print("error flag value")


class FogNode:
    def __init__(self, server: Server):
        self.server = server
        self.data = None

    def collect(self, ciphertexts: list):
        worker_num = ciphertexts.__len__()
        task_num = ciphertexts[0].__len__()

        ciphertexts_sum = []
        for j in range(task_num):
            s = ciphertexts[0][j]
            for i in range(1, worker_num):
                s += ciphertexts[i][j]
            ciphertexts_sum.append(s)
        return self.server.upload(ciphertexts_sum)

    def cal(self, ciphertexts: list):
        return self.server.cal(ciphertexts)

    def upload_sensory_data(self, flag: int, data: list, weights: list = None, adjs: list = None):
        return self.server.truth_discovery(flag, data, weights, adjs)


class Worker:
    def __init__(self, epsilon, omega, alpha1, w, raw_data):
        self.epsilon = epsilon
        self.alpha1 = alpha1
        self.omega = omega
        self.w = w
        self.raw_data = raw_data

        self.info = {
            "left_budget": epsilon,
            "cost_budgets": [],
            "crh_weights": [],
            "avg": 0,
            "variant": 0,
            "weight_error": [],
            "imp": 0.0,
            "credibility_weight": 0
        }

    # outlier-aware weight estimation mechanism
    def outlier_aware(self):
        last_variant = self.info["variant"]
        last_weight = self.info["crh_weights"][-1]
        # update attenuation variant
        self.info["avg"] = self.alpha1 * last_weight + (1 - self.alpha1) * self.info["avg"]
        self.info["variant"] = self.alpha1 * (self.info["avg"] - last_weight) ** 2 + (1 - self.alpha1) * self.info["variant"]
        d = self.info["variant"] - last_variant
        if d > self.info["beta1"]:
            window = min(self.info["crh_weights"].__len__(), self.w)
            credibility_weight = 0
            for i in range(2, window):
                # ARMA
                xi = (self.alpha1 * (1 - self.alpha1) ** (i-1))
                credibility_weight += xi * (self.info["crh_weights"][-i] + self.info["weight_error"][-i])
            credibility_weight += self.alpha1 * last_weight / 2
            self.info["credibility_weight"] = credibility_weight
        else:
            self.info["credibility_weight"] = last_weight
        return self.info["credibility_weight"]

    # weight-aware budget allocation mechanism
    def weight_aware(self):
        window = min(self.omega, self.info["crh_weights"].__len__())

        # remaining budget
        if self.info["cost_budgets"].__len__() > self.omega:
            self.info["left_budget"] += self.info["cost_budgets"][-self.omega]

        # compute importance
        sum_weight = self.info["credibility_weight"]
        for i in range(1, window):
            sum_weight += self.info["crh_weights"][-i]
        mu = self.info["credibility_weight"] / sum_weight

        # compute budget
        if self.info["left_budget"] * 2 > self.epsilon:
            p = 1 - 1 / (math.e ** mu)
        else:
            alpha2 = math.sin((2 * math.pi * (self.info["left_budget"] - self.epsilon / 4)) / self.epsilon) / 2 + 0.5
            delta = math.fabs(self.info["imp"] - mu)
            if delta == 0:
                delta = 0.00001
            x = alpha2 * mu + (1 - alpha2) / delta
            if x > 3:
                p = 0.8
            else:
                p = min(0.8, 1 - 1 / (math.e ** x))
        budget = self.info["left_budget"] * p
        self.info["cost_budgets"].append(budget)
        self.info["left_budget"] -= budget
        self.info["imp"] = mu
        return budget

    def first_upload_answers(self):
        self.info["cost_budgets"].append(0)
        return self.raw_data[0]

    def before_k_rounds(self, t):
        self.info["avg"] = self.alpha1 * self.info["crh_weights"][-1] + (1 - self.alpha1) * self.info["avg"]
        self.info["variant"] = self.alpha1 * (self.info["avg"] - self.info["crh_weights"][-1]) ** 2 + (1 - self.alpha1) * self.info["variant"]
        self.info["credibility_weight"] = self.info["crh_weights"][-1]
        self.info["cost_budgets"].append(0)
        if t == 10:
            window = min(self.omega, self.info["crh_weights"].__len__())
            self.info["imp"] = self.info["crh_weights"][-1] / sum(self.info["crh_weights"][-window:])
        return self.raw_data[t], self.info["credibility_weight"]

    def after_k_rounds(self, t):
        self.outlier_aware()
        budget = self.weight_aware()

        cur_data = self.raw_data[t]
        perturbed_data = []
        adj_array = []
        for d in cur_data:
            noise = math.fabs(np.random.laplace(0, 1 / budget))
            perturbed_data.append(d + noise)
            adj_array.append(-self.info["credibility_weight"] * noise / (d + noise))

        min_temp = 100000
        candidate = 0
        for adj_val in adj_array:
            total_temp = 0
            for i in range(cur_data.__len__()):
                total_temp += math.fabs((self.info["credibility_weight"] + adj_val) * perturbed_data[i] - self.info["credibility_weight"] * cur_data[i])
            if total_temp < min_temp:
                candidate = adj_val
                min_temp = total_temp
        return perturbed_data, self.info["credibility_weight"], candidate

    def update_weight(self, t, weight):
        self.info["weight_error"].append(weight - self.info["credibility_weight"])
        if t == 10:
            self.info["beta1"] = sum(self.info["weight_error"]) / (self.info["weight_error"].__len__() - 1)

        self.info["crh_weights"].append(weight)

    def get_data(self, t):
        return self.raw_data[t]


def weight_estimation(data: list, truths: list):
    """
    A clean version without encryption for easy experimentation.
    """
    m, n = data.__len__(), data[0].__len__()
    sum_distance = 0
    weights = []

    std = []
    for i in range(n):
        sum_i = 0
        for j in range(m):
            sum_i += data[j][i]
        avg_i = sum_i / m
        std_i = 0
        for j in range(m):
            std_i += (avg_i - data[j][i]) ** 2
        std.append(0.00001 if std_i == 0 else math.sqrt(std_i / m))

    for i in range(m):
        d_i = 0
        for j in range(n):
            d_i += ((truths[j] - data[i][j]) ** 2) / std[j]
        sum_distance += d_i
        weights.append(d_i)
    if sum_distance == 0:
        sum_distance = 0.00001

    for i in range(weights.__len__()):
        weights[i] = -math.log((weights[i] / sum_distance))
    return weights


def weight_estimation2(pk, node: FogNode, data: list, truths: list):
    """
    The normal version with encryption implemented according to the paper.
    Each encryption operation should be carried out independently on the terminals of different users,
    but the conditions are limited, and serial simulation is used here.
    """
    m = data.__len__()
    n = data[0].__len__()

    enc_data = []
    for i in range(m):
        temp = []
        for j in range(n):
            temp.append(pk.encrypt(data[i][j]))
        enc_data.append(temp)
    aggregated_results = node.collect(enc_data)
    avg = [s / m for s in aggregated_results]

    squared_answer = []
    for i in range(m):
        temp = []
        for j in range(n):
            temp.append(pk.encrypt((data[i][j] - avg[j]) ** 2))
        squared_answer.append(temp)
    aggregated_results = node.collect(squared_answer)
    stds = [math.sqrt(s / m) for s in aggregated_results]

    pow_answer = []
    encrypted_pow_answer = []
    for i in range(m):
        sum_answer = 0
        for j in range(n):
            if stds[j] != 0:
                sum_answer += (data[i][j] - truths[j]) ** 2 / stds[j]
        encrypted_pow_answer.append(pk.encrypt(sum_answer))
        pow_answer.append(sum_answer)

    loss = node.cal(encrypted_pow_answer)
    weights = [-math.log(pow_answer[i] / loss) for i in range(m)]
    return weights


def start(test_type):
    dataset_name = "weather"

    log_prefix = f".log/{dataset_name}"
    base_dataset = ".dataset/" + dataset_name

    if dataset_name == "lab":
        alpha_1 = 0.4
    elif dataset_name == "weather":
        alpha_1 = 0.2
    else:
        print("unknown dataset")
        return

    epsilon = 1
    omega = 10
    beta_2 = 0.01

    w = 0
    while alpha_1 * ((1 - alpha_1) ** w) >= beta_2:
        w += 1

    # Authority Center
    ac = Center()
    # Public and Private Key
    pk = ac.get_pub_key()
    sk = ac.get_pri_key()
    # Server
    server = Server(sk)
    # Fog Node
    node = FogNode(server)

    for r in range(100):
        worker_list = []
        for file_name in os.listdir(base_dataset):
            with open(f"{base_dataset}/{file_name}") as f:
                temp_list = []
                for line in f:
                    if base_dataset.endswith("lab"):
                        ele = line.strip().split(";")[1:]
                    else:
                        ele = line.strip().split(";")
                    answers = [math.fabs(float(x)) for x in ele]
                    temp_list.append(answers)
            worker_list.append(Worker(epsilon, omega, alpha_1, w, temp_list))

        t = worker_list[0].raw_data.__len__()

        log_file = open(f"{log_prefix}/{test_type}/1/{r}.txt", "w")
        for i in range(t):
            if i == 0:
                up_data = []
                for worker in worker_list:
                    up_data.append(worker.first_upload_answers())
                discovered_truths = node.upload_sensory_data(1, up_data)
            elif i <= 10:
                up_data = []
                up_weight = []
                for worker in worker_list:
                    data, weight = worker.before_k_rounds(i)
                    up_data.append(data)
                    up_weight.append(weight)
                discovered_truths = node.upload_sensory_data(2, up_data, up_weight)
            else:
                up_data = []
                up_weight = []
                up_adj = []
                for worker in worker_list:
                    data, weight, adj = worker.after_k_rounds(i)
                    up_data.append(data)
                    up_weight.append(weight)
                    up_adj.append(adj)
                discovered_truths = node.upload_sensory_data(3, up_data, up_weight, up_adj)

            log_file.write(discovered_truths.__str__() + "\n")

            raw_data = []
            for worker in worker_list:
                raw_data.append(worker.get_data(i))

            # weights = weight_estimation2(pk, node, raw_data, discovered_truths)
            weights = weight_estimation(raw_data, discovered_truths)
            for j in range(weights.__len__()):
                worker_list[j].update_weight(i, weights[j])
        log_file.close()


if __name__ == '__main__':
    start("epsilon")