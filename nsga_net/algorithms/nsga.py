"""
NSGA-II Multi-Objective Optimization Algorithm
"""

from typing import List


class NSGAII:
    """NSGA-II algorithm for multi-objective optimization"""
    
    @staticmethod
    def dominates(a1, a2, objs: List[str] = ['accuracy', 'flops']) -> bool:
        better = False
        for obj in objs:
            v1, v2 = a1.objectives.get(obj, 0), a2.objectives.get(obj, 0)
            if obj == 'accuracy':
                if v1 > v2: better = True
                elif v1 < v2: return False
            else:
                if v1 < v2: better = True
                elif v1 > v2: return False
        return better
    
    @staticmethod
    def non_dominated_sort(pop, objs: List[str] = ['accuracy', 'flops']):
        if not pop:
            return [[]]
        
        fronts = [[]]
        for p in pop:
            p.domination_count = 0
            p.dominated_solutions = []
        
        for i, p in enumerate(pop):
            for j, q in enumerate(pop):
                if i != j:
                    if NSGAII.dominates(p, q, objs):
                        p.dominated_solutions.append(q)
                    elif NSGAII.dominates(q, p, objs):
                        p.domination_count += 1
            if p.domination_count == 0:
                p.rank = 1
                fronts[0].append(p)
        
        if not fronts[0]:
            for p in pop:
                p.rank = 1
            fronts[0] = list(pop)
        
        i = 0
        while i < len(fronts) and len(fronts[i]) > 0:
            nf = []
            for p in fronts[i]:
                for q in p.dominated_solutions:
                    q.domination_count -= 1
                    if q.domination_count == 0:
                        q.rank = i + 2
                        nf.append(q)
            i += 1
            if nf:
                fronts.append(nf)
        
        while fronts and not fronts[-1]:
            fronts.pop()
        return fronts if fronts else [[]]
    
    @staticmethod
    def calculate_crowding_distance(front, objs: List[str] = ['accuracy', 'flops']):
        n = len(front)
        if n == 0:
            return
        for a in front:
            a.crowding_distance = 0
        if n < 2:
            if n == 1:
                front[0].crowding_distance = float('inf')
            return
        for obj in objs:
            front.sort(key=lambda x: x.objectives.get(obj, 0))
            front[0].crowding_distance = front[-1].crowding_distance = float('inf')
            obj_range = front[-1].objectives.get(obj, 0) - front[0].objectives.get(obj, 0)
            if obj_range > 0:
                for i in range(1, n - 1):
                    front[i].crowding_distance += (front[i+1].objectives.get(obj, 0) - front[i-1].objectives.get(obj, 0)) / obj_range
    
    @staticmethod
    def select_population(pop, size: int, objs: List[str] = ['accuracy', 'flops']):
        fronts = NSGAII.non_dominated_sort(pop, objs)
        new_pop = []
        for front in fronts:
            if len(new_pop) + len(front) <= size:
                new_pop.extend(front)
            else:
                NSGAII.calculate_crowding_distance(front, objs)
                front.sort(key=lambda x: x.crowding_distance, reverse=True)
                new_pop.extend(front[:size - len(new_pop)])
                break
        return new_pop
