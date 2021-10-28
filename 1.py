import random

from tensorflow import keras
from tensorflow.keras.layers import *
from DataHelper import *
from MyMetrics import *


DEBUG = False

def debug(*args, sep=' ', end='\n', file=None):
    global DEBUG
    if DEBUG is True:
        print(*args, sep=sep, end=end, file=file)



def print_data(y):
    d1 = {}
    for y1 in y:
        if y1 not in d1:
            d1[y1] = 0
        else:
            d1[y1] += 1
    print(d1)


def training_model():
    """
    根据 current_population 中的个体训练模型
    """
    print("[ training_model ]")
    global epochs, batch_size, x_val, y_val
    global intermediate_population_2, best_models, y_pred_2, intermediate_models_2

    # 根据种群训练模型
    y_pred_2.clear()
    intermediate_models_2.clear()
    train_acc = []
    val_acc = []
    for i, individual in enumerate(intermediate_population_2):
        debug(list(individual))
        x_train_selected, y_train_selected = get_selected_samples(individual)
        global DEBUG
        if DEBUG:
            print_data(y_train_selected)
        new_model = get_model()
        history = new_model.fit(x_train_selected, y_train_selected, batch_size=batch_size,
                                epochs=epochs, verbose=0)
        train_acc.append(history.history["acc"][-1])
        y_pred = new_model.predict(x_val)
        y_pred = np.argmax(y_pred, axis=1)
        y_pred_2.append(y_pred)
        acc1 = acc(y_val, y_pred)
        val_acc.append(acc1)

        intermediate_models_2.append(new_model)

    print("train_acc:", train_acc)
    print("val_acc:", val_acc)


def evaluation():
    """
    评估每个训练好的模型（PPV 和 PFC）
    """
    print("[ evaluation ]")
    global fitness_ppv_2, fitness_PFC_2, intermediate_gmean_2, intermediate_mAuc_2
    global y_pred_2, y_val
    global best_gmean, best_mAuc, best_y_pred, best_models
    global current_population, current_ppv, current_PFC, current_models, current_y_pred, current_y_pred
    global current_gmean, current_mAuc

    fitness_ppv_2.clear()
    intermediate_gmean_2.clear()
    intermediate_mAuc_2.clear()

    for y_pred in y_pred_2:
        p = ppv(y_val, y_pred)
        fitness_ppv_2.append(p)

        mAUC1 = mAUC(y_val, y_pred)
        intermediate_mAuc_2.append(mAUC1)

        gmean1 = gmean(y_val, y_pred)
        intermediate_gmean_2.append(gmean1)

    fitness_PFC_2 = PFC(y_val, y_pred_2)

    # 首次运行结果保存为最优
    if len(best_gmean) == 0:
        current_gmean = intermediate_gmean_2.copy()
        current_mAuc = intermediate_mAuc_2.copy()
        current_y_pred = y_pred_2.copy()
        current_models = intermediate_models_2.copy()
        current_ppv = fitness_ppv_2.copy()
        current_PFC = fitness_PFC_2.copy()
        current_population = intermediate_population_2.copy()

        best_gmean = intermediate_gmean_2.copy()
        best_mAuc = intermediate_mAuc_2.copy()
        best_y_pred = y_pred_2.copy()
        best_models = intermediate_models_2.copy()

    print("fitness_ppv_2:", fitness_ppv_2)
    print("fitness_PFC_2:", fitness_PFC_2)
    print("intermediate_gmean_2:", intermediate_gmean_2)
    print(min(intermediate_gmean_2))
    print("intermediate_mAuc_2:", intermediate_mAuc_2)
    print(min(intermediate_mAuc_2))


def select_new_population():
    """
    从父代和子代种群中择优挑选
    """
    print("[ \tselect_new_population ]")

    #  如果经过相似解消除只剩下 population_size 个个体了，就不用挑选了，没得选了
    global intermediate_population_2, current_population
    global current_ppv, current_PFC, fitness_ppv_2, fitness_PFC_2, current_gmean, current_mAuc, current_y_pred
    global intermediate_models_2, current_models
    global best_gmean, best_mAuc, best_models, best_y_pred, y_pred_2

    if len(intermediate_population_2) == population_size:
        current_population = intermediate_population_2.copy()
        current_ppv = fitness_ppv_2.copy()
        current_PFC = fitness_PFC_2.copy()
        current_gmean = intermediate_gmean_2.copy()
        current_mAuc = intermediate_mAuc_2.copy()
        current_models = intermediate_models_2.copy()
        current_y_pred = y_pred_2.copy()
        debug("current_population:")
        for x in current_population:
            debug(list(x))
    else:
        # 清空之前的数据
        current_ppv.clear()
        current_PFC.clear()
        current_gmean.clear()
        current_mAuc.clear()
        current_models.clear()
        current_y_pred.clear()

        # 选择新个体（快速非支配性排序）
        # 找出第 1 层级
        F = []  # 个体分层
        P = intermediate_population_2
        n_dominate = np.zeros(len(P), dtype="int32")
        rank = np.zeros(len(P), dtype="int32")
        F1 = []
        S_dominate = {}
        for i in range(len(P)):
            S_dominate[i] = []  # 个体 i 支配的集合
        for i in range(len(P)):
            for j in range(len(P)):
                p = (fitness_ppv_2[i], fitness_PFC_2[i])
                q = (fitness_ppv_2[j], fitness_PFC_2[j])
                if dominate(p, q):
                    S_dominate[i].append(j)
                elif dominate(q, p):
                    n_dominate[i] += 1  # 支配个体 i 的个体数量
            if n_dominate[i] == 0:
                rank[i] = 1
                F1.append(i)
        F.append(F1)

        # 再找出第 2,3, ... 层级
        i = 0
        while len(F[i]) != 0:
            Q = []
            for i_p in F[i]:
                for i_q in S_dominate[i_p]:
                    n_dominate[i_q] -= 1
                    if n_dominate[i_q] == 0:
                        rank[i_q] = i + 1
                        Q.append(i_q)
            i += 1
            F.append(Q)

        # 根据快速非支配排序结果挑选出新种群
        current_population.clear()
        for i in range(len(F)):
            for j in F[i]:
                if len(current_population) < population_size:
                    # 去掉 G-Mean 为 0 的个体
                    if intermediate_gmean_2[j] == 0:
                        continue
                    current_population.append(P[j])
                    current_ppv.append(fitness_ppv_2[j])
                    current_PFC.append(fitness_PFC_2[j])
                    current_gmean.append(intermediate_gmean_2[j])
                    current_mAuc.append(intermediate_mAuc_2[j])
                    current_models.append(intermediate_models_2[j])
                    current_y_pred.append(y_pred_2[j])
                else:
                    break
            if len(current_population) == population_size:
                break
        debug("current_population:")
        for x in current_population:
            debug(list(x))

    # 保存最优模型
    if min(current_gmean) > min(best_gmean) or min(current_mAuc) > min(best_mAuc):
        best_models = current_models.copy()
        best_gmean = current_gmean.copy()
        best_mAuc = current_mAuc.copy()
        best_y_pred = current_y_pred.copy()
    print("best gmean:", best_gmean)
    print(min(best_gmean))
    print("best mAUC:", best_mAuc)
    print(min(best_mAuc))

    # 计算投票结果
    calc_voting()

def calc_voting():
    global y_val, best_y_pred, y_true

    y_voting = []
    for j in range(len(best_y_pred[0])):
        y1 = [best_y_pred[i][j] for i in range(len(best_y_pred))]
        y11 = np.argmax(np.bincount(y1))
        y_voting.append(y11)

    print("voting acc:", acc(y_val, y_voting))
    print("voting gmean:", gmean(y_val, y_voting))
    print("voting mAUC:", mAUC(y_val, y_voting))


def dominate(p, q):
    p_ppv = p[0]
    p_PFC = p[1]

    q_ppv = q[0]
    q_PFC = q[1]

    n_equal = 0
    for key in p_ppv:
        if p_ppv[key] < q_ppv[key]:
            return False
        elif p_ppv[key] == q_ppv[key]:
            n_equal += 1

    if n_equal == len(p_ppv):
        if p_PFC > q_PFC:
            return True
        else:
            return False

    return True


def evolution():
    """
    个体进化
    """
    print("[ BEGIN evolution ]")

    parents = select_parents()
    childs = crossover(parents)
    mutation(childs)
    elimination()

    print("[ END evolution ]")


def crossover(parents):
    """
    一点交叉法，从亲本产生子代个体
    """
    print("[ \tcrossover ]")

    paired_parents_nums = int(len(parents) / 2)
    it = iter(parents)
    childs = []
    max_end_point = len(parents[0]) - int(len(parents[0])*0.1)
    for i in range(paired_parents_nums):
        father = next(it)
        mother = next(it)
        i_point = random.randint(0, max_end_point)
        # 【注意】两个父代个体经过交叉后生成两个子代个体
        child1 = np.concatenate((father[:i_point], mother[i_point:]))
        child2 = np.concatenate((mother[:i_point], father[i_point:]))
        childs.append(child1)
        childs.append(child2)

    # 单独处理只剩单个父代个体的情况：直接复制，变成子代
    if len(parents) % 2 == 1:
        childs.append(next(it))

    return childs


def mutation(childs):
    """
    子代个体发生变异
    """
    print("[ \tmutation ]")
    global intermediate_population_1

    mutation_childs = childs
    mutation_len = int(0.1 * len(childs[0]))  # 变异的长度
    n_mutation = int(0.1 * len(childs))
    index = random.sample(range(len(childs)), n_mutation)  # 10%的子代会变异
    for i in index:
        start_point = random.randint(0, len(childs[0]) - mutation_len - 1)
        for j in range(start_point, start_point + mutation_len):
            if childs[i][j] == 0:
                childs[i][j] = 1
            else:
                childs[i][j] = 0

    intermediate_population_1 = current_population + mutation_childs

    debug("intermediate_population_1:")
    for x in intermediate_population_1:
        debug(list(x))



def elimination():
    """
    消除过于相似的个体（相似度超过 xx% 即视为过于相似）
    """
    print("[ \telimination ]")
    global intermediate_population_1, intermediate_population_2, population_size

    # 设定相似删除阈值
    target_similar_score = 0.9
    intermediate_population_2.clear()
    n = len(intermediate_population_1)
    remove_index = np.zeros(n, dtype="int32")  # 移除索引，1 表示消除，0 表示保留
    remove_nums = 0
    remove_nums_expect = len(intermediate_population_1) - population_size

    # 找出过于相似的个体
    for i in range(n):
        if remove_index[i] == 1:
            continue
        for j in range(i + 1, n):
            if remove_index[j] == 1:
                continue
            similar_score = similar(intermediate_population_1[i], intermediate_population_1[j])
            debug(i, j, similar_score)
            if similar_score >= target_similar_score:
                remove_index[j] = 1  # 移除 j
                remove_nums += 1
                if remove_nums == remove_nums_expect:
                    break
        if remove_nums == remove_nums_expect:
            break

    # 保留剩下的个体到 intermediate_population_2
    debug("remove_index:", remove_index)
    for i, val in enumerate(remove_index):
        if val != 1:
            intermediate_population_2.append(intermediate_population_1[i])

    # 打印 intermediate_population_2
    debug("intermediate_population_2:")
    for x in intermediate_population_2:
        debug(list(x))


def similar(x1, x2):
    equal_nums = 0
    for i1, i2 in zip(x1, x2):
        if i1 == i2:
            equal_nums += 1
    similar_score = equal_nums / len(x1)

    return similar_score


def select_parents():
    """
    选择父代个体，用于交配产生子代

    :return:
    """
    print("[ \tselect_parents ]")
    global current_population, current_ppv, current_PFC, population_size

    # 竞标赛选择 parents
    parents_nums = population_size
    selected_parents = []
    for i in range(parents_nums):
        # 随机选 3 个，然后挑最好的一个
        index = random.sample(range(population_size), 3)
        choice_individual = [current_population[i] for i in index]
        choice_ppv = [current_ppv[i] for i in index]
        choice_PFC = [current_PFC[i] for i in index]
        best_individual = get_best_individual(choice_individual, choice_ppv, choice_PFC)
        selected_parents.append(best_individual)

    return selected_parents


def get_best_individual(choice_individual, choice_PPVs, choice_PFCs):
    """
    从给出的个体中挑选出最好的个体
    """
    if dominate((choice_PPVs[0], choice_PFCs[0]), (choice_PPVs[1], choice_PFCs[1])):
        if dominate((choice_PPVs[0], choice_PFCs[0]), (choice_PPVs[2], choice_PFCs[2])):
            i_best = 0
        else:
            i_best = 2
    else:
        if dominate((choice_PPVs[1], choice_PFCs[1]), (choice_PPVs[2], choice_PFCs[2])):
            i_best = 1
        else:
            i_best = 2

    return choice_individual[i_best]


def predict(x):
    y_pred = None
    for model in best_models:
        if y_pred is None:
            y_pred = model.predict(x)
            y_pred = np.array(y_pred)
        else:
            y_pred += np.array(model.predict(x))

    return y_pred


def get_model():
    inputs = keras.Input(shape=x_train[0].shape)
    flatten_1 = Flatten()(inputs)
    dense_1 = Dense(20, activation="relu")(flatten_1)
    global class_num
    outputs = Dense(class_num, activation="softmax")(dense_1)
    model = keras.Model(inputs, outputs)
    # model.summary()
    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(), optimizer="Adam", metrics=["acc"])

    return model


def get_individual():
    global train_data

    # 少数类最少样本数量
    min_num = None
    for key in train_data:
        if min_num is None:
            min_num = len(train_data[key])
        else:
            if min_num > len(train_data[key]):
                min_num = len(train_data[key])

    sampling_num_every_class = int(min_num * 0.9)
    new_individual = np.full(len(x_train), 0, dtype="int32")
    start = 0
    for key in train_data:
        index = random.sample(range(len(train_data[key])), sampling_num_every_class)
        for i in index:
            new_individual[i + start] = 1
        start += len(train_data[key])

    return new_individual


def get_selected_samples(individual):
    global x_train, y_train

    x_train_selected = []
    y_train_selected = []
    for i, val in enumerate(individual):
        if val == 1:
            x_train_selected.append(x_train[i])
            y_train_selected.append(y_train[i])

    return np.array(x_train_selected), np.array(y_train_selected)


def init_population():
    """
    初始化种群

    :return:
    """
    print("[ init_population ]")
    global population_size, intermediate_population_2
    global epochs

    for i in range(population_size):
        new_individual = get_individual()
        intermediate_population_2.append(new_individual)

    for x in intermediate_population_2:
        debug(list(x))


def arrange_data(x, y):
    """
    数据按类整理，存放到字典中

    :param x:
    :param y:
    :return:
    """
    x_data = {}
    for i, x_i in enumerate(x):
        if y[i] in x_data:
            x_data[y[i]].append(x_i)
        else:
            x_data[y[i]] = [x_i]

    for i, key in enumerate(x_data):
        if i == 0:
            x = np.array(x_data[key])
            y = np.full(len(x_data[key]), key, dtype="float32")
        else:
            x = np.vstack((x, x_data[key]))
            y1 = np.full(len(x_data[key]), key, dtype="float32")
            y = np.concatenate((y, y1))

    return x_data


# 获取数据
x, y = get_data("Car")

# 按类整理
data = arrange_data(x, y)
class_num = len(data)

# 按比例分割
x_train, y_train, x_val, y_val = split(data, 0.2)
print("x_train:", x_train.shape)
print_data(y_train)
train_data = arrange_data(x_train, y_train)

epochs = 100
batch_size = 64

population_size = 30  # 种群大小

best_models = []
best_gmean = []
best_mAuc = []
best_y_pred = []

intermediate_population_1 = []

intermediate_population_2 = []
intermediate_gmean_2 = []
intermediate_mAuc_2 = []
fitness_ppv_2 = []
fitness_PFC_2 = []
intermediate_models_2 = []
y_pred_2 = []

current_population = []
current_ppv = []
current_PFC = []
current_gmean = []
current_mAuc = []
current_models = []
current_y_pred = []

n_genetic = 100
for i in range(n_genetic):
    print("\n--------> n_genetic %d" % (i + 1))
    if i == 0:
        init_population()
        training_model()
        evaluation()
    evolution()
    training_model()
    evaluation()
    select_new_population()
