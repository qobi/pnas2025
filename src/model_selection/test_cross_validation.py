import torch

def validate_folds(folds, category, exemplar):
    
    idx = 0
    jdx = 0
    
    print(f"Category distribution: {category.unique(return_counts=True)}")
    print(f"Exemplar distribution: {exemplar.unique(return_counts=True)}")

    f = folds[idx]

    train = f['train'][jdx]
    train_category_dist = category[train].unique(return_counts=True)
    train_exemplar_dist = exemplar[train].unique(return_counts=True)
    
    val = f['val'][jdx]
    val_category_dist = category[val].unique(return_counts=True)
    val_exemplar_dist = exemplar[val].unique(return_counts=True)

    train_val = f['train_val']
    train_val_category_dist = category[train_val].unique(return_counts=True)
    train_val_exemplar_dist = exemplar[train_val].unique(return_counts=True)

    confounded_test = f['confounded_test']
    confounded_test_category_dist = category[confounded_test].unique(return_counts=True)
    confounded_test_exemplar_dist = exemplar[confounded_test].unique(return_counts=True)

    unconfounded_test = f['unconfounded_test']
    unconfounded_test_category_dist = category[unconfounded_test].unique(return_counts=True)
    unconfounded_test_exemplar_dist = exemplar[unconfounded_test].unique(return_counts=True)

    # 1. Test no partitions have any overlap
    assert torch.isin(train, val).sum() == 0, "Train and validation sets overlap!"
    assert torch.isin(train, confounded_test).sum() == 0, "Train and confounded test sets overlap!"
    assert torch.isin(train, unconfounded_test).sum() == 0, "Train and unconfounded test sets overlap!"
    assert torch.isin(val, confounded_test).sum() == 0, "Validation and confounded test sets overlap!"
    assert torch.isin(val, unconfounded_test).sum() == 0, "Validation and unconfounded test sets overlap!"

    print("Property 1: No overlap between partitions is satisfied.")

    assert torch.isin(train_exemplar_dist[0], unconfounded_test_exemplar_dist[0]).sum() == 0, "Seen exemplars in unconfounded test set!"
    assert torch.isin(val_exemplar_dist[0], unconfounded_test_exemplar_dist[0]).sum() == 0, "Seen exemplars in unconfounded test set!"
    
    print("Property 2: Exemplar exclusivity is satisfied.")

    assert torch.allclose(train_category_dist[1]/len(train), val_category_dist[1]/len(val)), "Category distribution in train and validation sets is not equal!"
    assert torch.allclose(train_category_dist[1]/len(train), confounded_test_category_dist[1]/len(confounded_test)), "Category distribution in train and confounded test sets is not equal!"
    assert torch.allclose(train_category_dist[1]/len(train), unconfounded_test_category_dist[1]/len(unconfounded_test)), "Category distribution in train and unconfounded test sets is not equal!"
    assert torch.allclose(val_category_dist[1]/len(val), confounded_test_category_dist[1]/len(confounded_test)), "Category distribution in validation and confounded test sets is not equal!"
    assert torch.allclose(val_category_dist[1]/len(val), unconfounded_test_category_dist[1]/len(unconfounded_test)), "Category distribution in validation and unconfounded test sets is not equal!"

    print("Property 3: Category distribution is equal across partitions.")

    assert torch.allclose(train_exemplar_dist[1]/len(train), val_exemplar_dist[1]/len(val)), "Exemplar distribution in train and validation sets is not equal!"
    assert torch.allclose(train_exemplar_dist[1]/len(train), confounded_test_exemplar_dist[1]/len(confounded_test)), "Exemplar distribution in train and confounded test sets is not equal!"

    print("Property 4: Exemplar distribution is equal across train, validation, and confounded test sets.")