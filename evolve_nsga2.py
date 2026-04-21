import random
import numpy as np
from deap import base, creator, tools
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. 定義基因與「多目標」遺傳規則
# ==========================================
# 基因[0]: 用哪條短期均線? (0: 5MA, 1: 10MA, 2: 20MA)
# 基因[1]: 乖離率要在什麼區間? (0: 0~5%, 1: 5~10%, 2: 不限)
# 基因[2]: MACD條件? (0: 要大於0, 1: 不限)
GENE_LENGTH = 3

# 🔥 核心升級：定義多目標 (Multi-Objective)
# weights=(1.0, -1.0) 代表：第一個目標求「最大化(報酬)」，第二個目標求「最小化(MDD)」
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("attr_int", random.randint, 0, 2)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, GENE_LENGTH)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# ==========================================
# 2. 定義適應度函數 (考試卷：模擬回測，產出 報酬 與 MDD)
# ==========================================
def evaluate_strategy(individual):
    """
    實戰中這裡會呼叫 Backtrader 回傳 (總報酬率, 最大回撤MDD)
    這裡用模擬的方式來展示 NSGA-II 尋找 Pareto Front 的能力。
    """
    expected_return = 0
    expected_mdd = 0
    
    # 不同的均線選擇，會有不同的風險與報酬特性
    if individual[0] == 0:   # 5MA：報酬高，但容易被洗盤(MDD大)
        expected_return += 60
        expected_mdd += 25
    elif individual[0] == 1: # 10MA：平衡
        expected_return += 45
        expected_mdd += 15
    elif individual[0] == 2: # 20MA：穩健，報酬中等，MDD小
        expected_return += 35
        expected_mdd += 8
        
    if individual[1] == 0:   # 乖離小：穩健
        expected_return += 20
        expected_mdd -= 5
    if individual[2] == 0:   # MACD>0：動能強
        expected_return += 15
        expected_mdd += 2
        
    # 加入隨機雜訊模擬真實市場
    expected_return += random.uniform(-5, 5)
    expected_mdd += random.uniform(-2, 2)
    
    # 回傳 Tuple: (報酬率, MDD)
    return (expected_return, max(0, expected_mdd))

toolbox.register("evaluate", evaluate_strategy)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=2, indpb=0.3)
# 🔥 核心升級：選擇法改用 NSGA-II 專屬的非主導排序法
toolbox.register("select", tools.selNSGA2)

# ==========================================
# 3. 啟動 NSGA-II 演化引擎
# ==========================================
def run_nsga2_evolution():
    print("🧬 啟動 NSGA-II (非主導排序多目標遺傳演算法)...")
    MU = 50      # 種群數量
    NGEN = 10    # 演化代數
    CXPB = 0.7   # 交配率
    MUTPB = 0.2  # 突變率

    pop = toolbox.population(n=MU)
    
    # 評估初始種群
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # 必須先對初始種群進行 NSGA-II 的擁擠度分配 (Crowding Distance)
    pop = toolbox.select(pop, len(pop))

    print("\n⚔️ 開始多目標物競天擇 (Evolution)...")
    for gen in range(1, NGEN + 1):
        # 產生子代 (使用 NSGA-II 推薦的 TournamentDCD)
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        # 交配與突變
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)
                del ind1.fitness.values, ind2.fitness.values
        
        for ind in offspring:
            if random.random() <= MUTPB:
                toolbox.mutate(ind)
                del ind.fitness.values

        # 評估需要重新計算的子代
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # 將父代與子代合併後，用 NSGA-II 挑出最強的下一代
        pop = toolbox.select(pop + offspring, MU)
        if gen % 2 == 0:
            print(f" └ 第 {gen}/{NGEN} 代演化完成...")

    # ==========================================
    # 4. 解析 Pareto Front (帕雷托前緣)
    # ==========================================
    # 帕雷托前緣上的策略，代表在「相同的MDD下，報酬最高」或「相同的報酬下，MDD最低」
    front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
    
    # 根據風險(MDD)由小到大排序
    front.sort(key=lambda x: x.fitness.values[1])
    
    ma_map = {0: '5MA(激進)', 1: '10MA(平衡)', 2: '20MA(防守)'}
    
    print("\n" + "="*50)
    print("👑 NSGA-II 演化完成！找到最優策略組合 (Pareto Front)：")
    print("="*50)
    print(f"{'策略代碼':<10} | {'預期報酬(大越好)':<15} | {'最大回撤(小越好)':<15} | {'核心均線'}")
    print("-" * 50)
    
    for ind in front:
        ret = ind.fitness.values[0]
        mdd = ind.fitness.values[1]
        ma_str = ma_map[ind[0]]
        print(f"{str(ind):<10} | {ret:>11.2f}%    | {mdd:>11.2f}%    | {ma_str}")
        
    print("="*50)
    print("💡 實戰建議：\n如果你是保守型玩家，請挑選上方 MDD 最小的參數寫入系統；\n如果你是積極型玩家，請挑選下方報酬率最高的參數。")

if __name__ == "__main__":
    run_nsga2_evolution()

