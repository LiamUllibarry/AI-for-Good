def test_load():
  return 'loaded'

def compute_probs(neg,pos):
  p0 = neg/(neg+pos)
  p1 = pos/(neg+pos)
  return [p0,p1]

def cond_prob(table, evidence, evidence_value, target, target_value):
  t_subset = up_table_subset(table, target, 'equals', target_value)
  e_list = up_get_column(t_subset, evidence)
  p_b_a = sum([1 if v==evidence_value else 0 for v in e_list])/len(e_list)
  return p_b_a + .01

def cond_probs_product(table,evidence_row,target,target_value):
  table_columns = up_list_column_names(table) #your function body below
  evidence_columns = table_columns[:-1]
  evidence_complete = up_zip_lists(evidence_columns, evidence_row)
  cond_prob_list = [cond_prob(table, evidence_column, evidence_value, target, target_value) for evidence_column, evidence_value in evidence_complete]
  partial_numerator = up_product(cond_prob_list)
  return partial_numerator

def prior_prob(table, target, target_value):
  t_list = up_get_column(table, target)
  p_a = sum([1 if v==target_value else 0 for v in t_list])/len(t_list)
  return p_a

def naive_bayes(table, evidence_row, target):
  #compute P(target=0|...) by using cond_prob_product, take the product of the list, finally multiply by P(target=0) using cond_prob
  neg_1 = cond_probs_product(table, evidence_row, target, 0) * prior_prob(table, target, 0)

  #do same for P(target=1|...)
  pos_1 = cond_probs_product(table, evidence_row, target, 1) * prior_prob(table, target, 1)

  #Use compute_probs to get 2 probabilities
  neg,pos = compute_probs(neg_1, pos_1)
  #return your 2 results in a list
  return [neg, pos]

def metrics(zipped_list):
  #asserts here
  assert isinstance(zipped_list, list), 'zipped_list should be a list'
  assert all(isinstance(item, list) for item in zipped_list), 'zipped_list should be a list of lists'
  assert all(len(item) == 2 for item in zipped_list), 'zipped_list should be a zipped list of pairs'
  #assert all(isinstance(pair, list) and len(pair) >= 2 and all(isinstance(value, int) and value >= 0 for value in pair) for pair in zipped_list), 'zipped_list should be a zipped list where each pair contains two non-negative integers'
  #body of function below
  tn = sum([1 if pair==[0,0] else 0 for pair in zipped_list])
  tp = sum([1 if pair==[1,1] else 0 for pair in zipped_list])
  fp = sum([1 if pair==[1,0] else 0 for pair in zipped_list])
  fn = sum([1 if pair==[0,1] else 0 for pair in zipped_list]) 

  #now can compute precicision, recall, f1, accuracy. Watch for divide by 0.
  precision= tp/(tp+fp) if (tp + fp)!=0 else 0
  recall = tp/(tp+fn) if (tp + fn)!=0 else 0
  accuracy = (tp + tn)/(tp + tn + fp + fn) if (tp + tn + fp + fn)!=0 else 0
  f1 = 2*((precision * recall)/(precision + recall)) if (precision + recall)!=0 else 0
  #now build dictionary with the 4 measures
  measures_dict = {'Accuracy':accuracy, 'F1':f1, 'Precision':precision, 'Recall':recall}
  #finally, return the dictionary
  return measures_dict
