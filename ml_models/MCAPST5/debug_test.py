#!/usr/bin/env python3
import numpy as np
import h5py
import torch
from transformers import T5EncoderModel, T5Tokenizer
import tensorflow as tf
from tensorflow.keras.models import Model
from xgboost import XGBClassifier
from pathlib import Path

# Constants
SEQ_LEN = 1200
EMB_DIM = 1024
BATCH_SIZE = 8

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
CHECKPOINT_MCAPST5 = CHECKPOINT_DIR / "mcapst5_pan_epoch_20.hdf5"
CHECKPOINT_XGB = CHECKPOINT_DIR / "xgboost_pan_epoch_20.bin"

def test_models():
    print("=== Testing Model Loading ===")
    
    # Test 1: Load XGBoost model
    try:
        xgb_model = XGBClassifier()
        xgb_model.load_model(str(CHECKPOINT_XGB))
        print("✓ XGBoost model loaded successfully")
        
        # Check model parameters
        print(f"XGBoost n_estimators: {xgb_model.n_estimators}")
        print(f"XGBoost n_classes_: {xgb_model.n_classes_}")
        
    except Exception as e:
        print(f"✗ Failed to load XGBoost model: {e}")
        return
    
    # Test 2: Load Keras model
    try:
        keras_model = tf.keras.models.load_model(str(CHECKPOINT_MCAPST5))
        print("✓ Keras model loaded successfully")
        print(f"Keras model layers: {len(keras_model.layers)}")
        
        # Check input/output shapes
        print(f"Input shape: {keras_model.input_shape}")
        print(f"Output shape: {keras_model.output_shape}")
        
        # Get intermediate layer
        intermediate = Model(inputs=keras_model.input, outputs=keras_model.layers[-2].output)
        print(f"Intermediate layer output shape: {intermediate.output_shape}")
        
    except Exception as e:
        print(f"✗ Failed to load Keras model: {e}")
        return
    
    # Test 3: Test with different types of dummy data
    print("\n=== Testing with Different Dummy Data ===")
    
    # Test case 1: Random data
    print("Test 1: Random data")
    dummy_x1 = np.random.random((1, SEQ_LEN, EMB_DIM)).astype(np.float16)
    dummy_x2 = np.random.random((1, SEQ_LEN, EMB_DIM)).astype(np.float16)
    
    try:
        feats = intermediate.predict([dummy_x1, dummy_x2], batch_size=1, verbose=0)
        preds = xgb_model.predict_proba(feats)
        print(f"Random data prediction: {preds[0, 1]:.6f}")
    except Exception as e:
        print(f"Failed: {e}")
    
    # Test case 2: All zeros
    print("Test 2: All zeros")
    dummy_x1 = np.zeros((1, SEQ_LEN, EMB_DIM), dtype=np.float16)
    dummy_x2 = np.zeros((1, SEQ_LEN, EMB_DIM), dtype=np.float16)
    
    try:
        feats = intermediate.predict([dummy_x1, dummy_x2], batch_size=1, verbose=0)
        preds = xgb_model.predict_proba(feats)
        print(f"All zeros prediction: {preds[0, 1]:.6f}")
    except Exception as e:
        print(f"Failed: {e}")
    
    # Test case 3: All ones
    print("Test 3: All ones")
    dummy_x1 = np.ones((1, SEQ_LEN, EMB_DIM), dtype=np.float16)
    dummy_x2 = np.ones((1, SEQ_LEN, EMB_DIM), dtype=np.float16)
    
    try:
        feats = intermediate.predict([dummy_x1, dummy_x2], batch_size=1, verbose=0)
        preds = xgb_model.predict_proba(feats)
        print(f"All ones prediction: {preds[0, 1]:.6f}")
    except Exception as e:
        print(f"Failed: {e}")
    
    # Test case 4: Different sequences
    print("Test 4: Different sequences")
    dummy_x1 = np.random.random((1, SEQ_LEN, EMB_DIM)).astype(np.float16)
    dummy_x2 = np.random.random((1, SEQ_LEN, EMB_DIM)).astype(np.float16) * 2  # Different scale
    
    try:
        feats = intermediate.predict([dummy_x1, dummy_x2], batch_size=1, verbose=0)
        preds = xgb_model.predict_proba(feats)
        print(f"Different sequences prediction: {preds[0, 1]:.6f}")
    except Exception as e:
        print(f"Failed: {e}")
    
    # Test case 5: Multiple samples
    print("Test 5: Multiple samples")
    dummy_x1 = np.random.random((5, SEQ_LEN, EMB_DIM)).astype(np.float16)
    dummy_x2 = np.random.random((5, SEQ_LEN, EMB_DIM)).astype(np.float16)
    
    try:
        feats = intermediate.predict([dummy_x1, dummy_x2], batch_size=1, verbose=0)
        preds = xgb_model.predict_proba(feats)
        print(f"Multiple samples predictions: {[f'{p:.6f}' for p in preds[:, 1]]}")
        
        # Check variance
        unique_preds = np.unique(preds[:, 1])
        print(f"Unique predictions: {len(unique_preds)}")
        if len(unique_preds) == 1:
            print("⚠️  WARNING: All predictions are identical - model may be broken!")
        else:
            print(f"Prediction variance: {np.var(preds[:, 1]):.8f}")
            
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    test_models()
