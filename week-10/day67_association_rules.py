"""
Day 67: Association Models
Topic: Apriori, FP-Growth, support, confidence, lift
"""
import numpy as np
import pandas as pd
from itertools import combinations
from collections import defaultdict

# --- Generate synthetic transaction data ---
np.random.seed(42)
items = ['Milk', 'Bread', 'Eggs', 'Butter', 'Cheese', 'Yogurt', 'Apples', 'Bananas']
n_transactions = 200
transactions = []
for _ in range(n_transactions):
    n_items = np.random.randint(1, 6)
    cart = list(np.random.choice(items, size=n_items, replace=False))
    transactions.append(cart)

print(f"Transactions: {len(transactions)}")
print(f"Sample transactions: {transactions[:3]}")

# --- Apriori from scratch ---
def apriori(transactions, min_support=0.05):
    n = len(transactions)
    item_counts = defaultdict(int)
    for t in transactions:
        for item in set(t):
            item_counts[frozenset([item])] += 1
    
    frequent = {k: v/n for k, v in item_counts.items() if v/n >= min_support}
    all_frequent = dict(frequent)
    k = 2
    while frequent:
        candidates = set()
        items_set = set()
        for f in frequent:
            for item in f:
                items_set.add(item)
        for combo in combinations(items_set, k):
            candidates.add(frozenset(combo))
        
        candidate_counts = defaultdict(int)
        for t in transactions:
            t_set = set(t)
            for c in candidates:
                if c.issubset(t_set):
                    candidate_counts[c] += 1
        
        frequent = {c: v/n for c, v in candidate_counts.items() if v/n >= min_support}
        all_frequent.update(frequent)
        k += 1
    return all_frequent

frequent_items = apriori(transactions, min_support=0.05)
print(f"\nFrequent Itemsets (min_support=0.05):")
for itemset, support in sorted(frequent_items.items(), key=lambda x: -x[1])[:10]:
    print(f"  {set(itemset)} -> support: {support:.3f}")

# --- Generate association rules ---
def generate_rules(frequent_items, min_confidence=0.3):
    rules = []
    for itemset, support in frequent_items.items():
        if len(itemset) < 2:
            continue
        for n in range(1, len(itemset)):
            for antecedent in combinations(itemset, n):
                antecedent = frozenset(antecedent)
                consequent = itemset - antecedent
                if antecedent in frequent_items and consequent:
                    confidence = support / frequent_items[antecedent]
                    if confidence >= min_confidence:
                        lift = confidence / frequent_items.get(consequent, support)
                        rules.append((set(antecedent), set(consequent), support, confidence, lift))
    return rules

rules = generate_rules(frequent_items, min_confidence=0.3)
print(f"\nAssociation Rules (min_conf=0.3):")
print(f"{'Antecedent':>15} -> {'Consequent':<15} {'Support':>8} {'Confidence':>10} {'Lift':>6}")
print("-" * 65)
for ant, cons, sup, conf, lift in sorted(rules, key=lambda x: -x[3])[:10]:
    print(f"{str(ant):>15} -> {str(cons):<15} {sup:8.3f} {conf:10.3f} {lift:6.3f}")

print("\nKey Concepts:")
print("- Support: P(A and B) = frequency of itemset / total transactions")
print("- Confidence: P(B|A) = support(A,B) / support(A)")
print("- Lift: P(B|A) / P(B) = confidence / support(B)")
print("- Lift > 1: positive correlation, Lift = 1: independent, Lift < 1: negative")
print("- Apriori: level-wise search, uses support pruning (downward closure)")
