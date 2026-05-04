import random
import numpy as np
from deap import base, creator, tools
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. Define genes and multi-objective rules
# ==========================================
# Gene[0]: short MA choice (0: 5MA, 1: 10MA, 2: 20MA).
# Gene[1]: bias range (0: 0-5%, 1: 5-10%, 2: any).
# Gene[2]: MACD rule (0: above 0, 1: any).
GENE_LENGTH = 3

# Define two goals: maximize return and minimize MDD.
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("attr_int", random.randint, 0, 2)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, GENE_LENGTH)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# ==========================================
# 2. Define the fitness function
# ==========================================
def evaluate_strategy(individual):
    """
    In production, call Backtrader and return total return and max drawdown.
    This mock shows how NSGA-II finds the Pareto front.
    """
    expected_return = 0
    expected_mdd = 0
    
    # Different MAs imply different risk and return profiles.
    if individual[0] == 0:   # 5MA: high return, high MDD.
        expected_return += 60
        expected_mdd += 25
    elif individual[0] == 1: # 10MA: balanced.
        expected_return += 45
        expected_mdd += 15
    elif individual[0] == 2: # 20MA: steadier, lower MDD.
        expected_return += 35
        expected_mdd += 8
        
    if individual[1] == 0:   # Low bias: steadier.
        expected_return += 20
        expected_mdd -= 5
    if individual[2] == 0:   # MACD > 0: stronger momentum.
        expected_return += 15
        expected_mdd += 2
        
    # Add noise to mimic real markets.
    expected_return += random.uniform(-5, 5)
    expected_mdd += random.uniform(-2, 2)
    
    # Return tuple: (return, MDD).
    return (expected_return, max(0, expected_mdd))

toolbox.register("evaluate", evaluate_strategy)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=2, indpb=0.3)
# Use NSGA-II non-dominated sorting.
toolbox.register("select", tools.selNSGA2)

# ==========================================
# 3. Run the NSGA-II engine
# ==========================================
def run_nsga2_evolution():
    print("🧬 啟動 NSGA-II (非主導排序多目標遺傳演算法)...")
    MU = 50      # Population size.
    NGEN = 10    # Number of generations.
    CXPB = 0.7   # Crossover probability.
    MUTPB = 0.2  # Mutation probability.

    pop = toolbox.population(n=MU)
    
    # Evaluate the initial population.
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Assign crowding distance before TournamentDCD.
    pop = toolbox.select(pop, len(pop))

    print("\n⚔️ 開始多目標物競天擇 (Evolution)...")
    for gen in range(1, NGEN + 1):
        # Create offspring with TournamentDCD.
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        # Apply crossover and mutation.
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)
                del ind1.fitness.values, ind2.fitness.values
        
        for ind in offspring:
            if random.random() <= MUTPB:
                toolbox.mutate(ind)
                del ind.fitness.values

        # Evaluate updated offspring.
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation from parents and offspring.
        pop = toolbox.select(pop + offspring, MU)
        if gen % 2 == 0:
            print(f" └ 第 {gen}/{NGEN} 代演化完成...")

    # ==========================================
    # 4. Parse the Pareto front
    # ==========================================
    # The front keeps the best return-risk tradeoffs.
    front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
    
    # Sort by risk from low to high.
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
