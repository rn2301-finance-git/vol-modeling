# evaluation/eval_phases.py
#
# This file defines hyperparameter tuning "phases" for different model types.
# In particular, it now includes dedicated phase configurations for the
# three‐headed transformer (under key "TRANSFORMER") and for XGBoost.
#
# You can modify the parameters (e.g. hidden_dim, nhead, num_layers, dropout,
# learning_rate, batch_size, etc.) as needed to explore different settings.


TEMP_PHASES = {
    "TRANSFORMER": {
        "PHASE1_GAMMA": [
            {
                "description": "Run 1A: Gamma 0.1",
                "hidden_dim": 64,
                "nhead": 4,
                "num_layers": 2,
                "dropout": 0.4,
                "learning_rate": 1e-4,
                "batch_size": 64,
                "gamma": 0.1,
                "weight_decay": 0.01,
                "use_parameter_groups": True,
                "gradient_accumulation_steps": 1,
                "eval_train_set_frequency": 5,
                "epochs": 100,
                "sequence_params": {
                    "sequence_length": 30,
                    "sample_every": 10
                }
            },
            {
                "description": "Run 1B: Gamma 0.2",
                "hidden_dim": 64,
                "nhead": 4,
                "num_layers": 2,
                "dropout": 0.4,
                "learning_rate": 1e-4,
                "batch_size": 64,
                "gamma": 0.2,
                "weight_decay": 0.01,
                "use_parameter_groups": True,
                "gradient_accumulation_steps": 1,
                "eval_train_set_frequency": 5,
                "epochs": 100,
                "sequence_params": {
                    "sequence_length": 30,
                    "sample_every": 10
                }
            },
            {
                "description": "Run 1C: Gamma 0.3",
                "hidden_dim": 64,
                "nhead": 4,
                "num_layers": 2,
                "dropout": 0.4,
                "learning_rate": 1e-4,
                "batch_size": 64,
                "gamma": 0.3,
                "weight_decay": 0.01,
                "use_parameter_groups": True,
                "gradient_accumulation_steps": 1,
                "eval_train_set_frequency": 5,
                "epochs": 100,
                "sequence_params": {
                    "sequence_length": 30,
                    "sample_every": 10
                }
            }
        ]
    }
}

PHASES = {
    "TRANSFORMER": {
        "PHASE1_BASELINE": [
            {
                "description": "Run 1: Transformer Baseline",
                "hidden_dim": 256,
                "nhead": 8,
                "num_layers": 3,
                "dropout": 0.1,
                "learning_rate": 1e-5,
                "batch_size": 64,
                "gradient_accumulation_steps": 1,
                "eval_train_set_frequency": 5,
                "epochs_fraction": 0.5,
                "sequence_params": {
                    "sequence_length": 30,
                    "sample_every": 10
                }
            }
        ],
        "PHASE2_ARCHITECTURE": [
            {
                "description": "Run 2A: Wider Transformer",
                "hidden_dim": 512,
                "nhead": 8,
                "num_layers": 3,
                "dropout": 0.1,
                "learning_rate": 1e-5,
                "batch_size": 64,
                "gradient_accumulation_steps": 1,
                "eval_train_set_frequency": 5,
                "epochs_fraction": 0.75,
                "sequence_params": {
                    "sequence_length": 30,
                    "sample_every": 10
                }
            },
            {
                "description": "Run 2B: Deeper Transformer",
                "hidden_dim": 256,
                "nhead": 8,
                "num_layers": 4,
                "dropout": 0.1,
                "learning_rate": 1e-5,
                "batch_size": 64,
                "gradient_accumulation_steps": 1,
                "eval_train_set_frequency": 5,
                "epochs_fraction": 0.75,
                "sequence_params": {
                    "sequence_length": 30,
                    "sample_every": 10
                }
            }
        ],
        "PHASE3_LEARNING": [
            {
                "description": "Run 3A: Higher Learning Rate",
                "hidden_dim": 256,
                "nhead": 8,
                "num_layers": 3,
                "dropout": 0.1,
                "learning_rate": 5e-5,
                "batch_size": 64,
                "gradient_accumulation_steps": 1,
                "eval_train_set_frequency": 5,
                "epochs_fraction": 0.75,
                "sequence_params": {
                    "sequence_length": 30,
                    "sample_every": 10
                }
            },
            {
                "description": "Run 3B: Lower Learning Rate",
                "hidden_dim": 256,
                "nhead": 8,
                "num_layers": 3,
                "dropout": 0.1,
                "learning_rate": 5e-6,
                "batch_size": 64,
                "gradient_accumulation_steps": 1,
                "eval_train_set_frequency": 5,
                "epochs_fraction": 0.75,
                "sequence_params": {
                    "sequence_length": 30,
                    "sample_every": 10
                }
            }
        ],
        "PHASE4_REGULARIZATION": [
            {
                "description": "Run 4A: Higher Dropout",
                "hidden_dim": 256,
                "nhead": 8,
                "num_layers": 3,
                "dropout": 0.2,
                "learning_rate": 1e-5,
                "batch_size": 64,
                "gradient_accumulation_steps": 1,
                "eval_train_set_frequency": 5,
                "epochs_fraction": 0.75,
                "sequence_params": {
                    "sequence_length": 30,
                    "sample_every": 10
                }
            },
            {
                "description": "Run 4B: Lower Dropout",
                "hidden_dim": 256,
                "nhead": 8,
                "num_layers": 3,
                "dropout": 0.05,
                "learning_rate": 1e-5,
                "batch_size": 64,
                "gradient_accumulation_steps": 1,
                "eval_train_set_frequency": 5,
                "epochs_fraction": 0.75,
                "sequence_params": {
                    "sequence_length": 30,
                    "sample_every": 10
                }
            }
        ],
        "PHASE5_FINAL": [
            {
                "description": "Run 5: Final Best Config",
                "hidden_dim": None,  # To be filled with the best configuration from previous phases
                "nhead": None,
                "num_layers": None,
                "dropout": None,
                "learning_rate": None,
                "batch_size": 64,
                "gradient_accumulation_steps": 1,
                "eval_train_set_frequency": 5,
                "epochs_fraction": 1.0,
                "sequence_params": {
                    "sequence_length": 30,
                    "sample_every": 10
                }
            }
        ]
    },

    "XGBOOST": {
        # XGBoost hyperparameter phases remain similar to the previous setup.
        "PHASE1_BASELINE": [
            {
                "description": "Run 1: XGB Baseline",
                "n_estimators": 200,
                "learning_rate": 0.1,
                "max_depth": 6,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.0,
                "reg_lambda": 1.0,
                "gamma": 0.0
            }
        ],
        "PHASE2_LR_TEST": [
            {
                "description": "Run 2A: LR=0.05",
                "n_estimators": 200,
                "learning_rate": 0.05,
                "max_depth": 6,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.0,
                "reg_lambda": 1.0,
                "gamma": 0.0
            },
            {
                "description": "Run 2B: LR=0.01",
                "n_estimators": 200,
                "learning_rate": 0.01,
                "max_depth": 6,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.0,
                "reg_lambda": 1.0,
                "gamma": 0.0
            }
        ],
        "PHASE3_MAX_DEPTH": [
            {
                "description": "Run 3A: max_depth=4",
                "n_estimators": 200,
                "learning_rate": None,  # Inherit best LR from PHASE2
                "max_depth": 4,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": None,
                "reg_lambda": None,
                "gamma": None
            },
            {
                "description": "Run 3B: max_depth=8",
                "n_estimators": 200,
                "learning_rate": None,
                "max_depth": 8,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": None,
                "reg_lambda": None,
                "gamma": None
            }
        ],
        "PHASE4_REGULARIZATION": [
            {
                "description": "Run 4A: L1 Regularization",
                "n_estimators": 200,
                "learning_rate": None,
                "max_depth": None,
                "subsample": None,
                "colsample_bytree": None,
                "reg_alpha": 0.1,
                "reg_lambda": None,
                "gamma": None
            },
            {
                "description": "Run 4B: L2 Regularization",
                "n_estimators": 200,
                "learning_rate": None,
                "max_depth": None,
                "subsample": None,
                "colsample_bytree": None,
                "reg_alpha": None,
                "reg_lambda": 1.5,
                "gamma": None
            },
            {
                "description": "Run 4C: Increase Gamma",
                "n_estimators": 200,
                "learning_rate": None,
                "max_depth": None,
                "subsample": None,
                "colsample_bytree": None,
                "reg_alpha": None,
                "reg_lambda": None,
                "gamma": 0.1
            }
        ],
        "PHASE5_SUBSAMPLE": [
            {
                "description": "Run 5A: subsample=0.6",
                "n_estimators": 200,
                "learning_rate": None,
                "max_depth": None,
                "subsample": 0.6,
                "colsample_bytree": 0.8,
                "reg_alpha": None,
                "reg_lambda": None,
                "gamma": None
            },
            {
                "description": "Run 5B: subsample=1.0",
                "n_estimators": 200,
                "learning_rate": None,
                "max_depth": None,
                "subsample": 1.0,
                "colsample_bytree": 0.8,
                "reg_alpha": None,
                "reg_lambda": None,
                "gamma": None
            }
        ],
        "PHASE6_FINAL": [
            {
                "description": "Run 6: Final Best XGB Config",
                "n_estimators": None,
                "learning_rate": None,
                "max_depth": None,
                "subsample": None,
                "colsample_bytree": 0.8,
                "reg_alpha": None,
                "reg_lambda": None,
                "gamma": None
            }
        ]
    },

    "SEQ_MLP": {
        "PHASE1_BASELINE": [
            {
                "description": "Run 1: Sequence MLP Baseline",
                "hidden_dim": 64,
                "dropout": 0.2,
                "learning_rate": 1e-4,
                "batch_size": 64,
                "gradient_accumulation_steps": 1,
                "eval_train_set_frequency": 5,
                "epochs": 50,
                "sequence_params": {
                    "short_seq_len": 20,
                    "long_seq_len": 60,
                    "sample_every": 10
                }
            }
        ],
        "PHASE2_ARCHITECTURE": [
            {
                "description": "Run 2A: Wider Network",
                "hidden_dim": 128,
                "dropout": 0.2,
                "learning_rate": 1e-4,
                "batch_size": 64,
                "gradient_accumulation_steps": 1,
                "eval_train_set_frequency": 5,
                "epochs": 50,
                "sequence_params": {
                    "short_seq_len": 20,
                    "long_seq_len": 60,
                    "sample_every": 10
                }
            },
            {
                "description": "Run 2B: Deeper Network",
                "hidden_dim": 64,
                "dropout": 0.2,
                "learning_rate": 1e-4,
                "batch_size": 64,
                "gradient_accumulation_steps": 2,
                "eval_train_set_frequency": 5,
                "epochs": 50,
                "sequence_params": {
                    "short_seq_len": 30,
                    "long_seq_len": 90,
                    "sample_every": 10
                }
            }
        ],
        "PHASE3_SEQUENCE": [
            {
                "description": "Run 3A: Longer Sequences",
                "hidden_dim": None,  # Will use best from previous phases
                "dropout": 0.2,
                "learning_rate": None,  # Will use best from previous phases
                "batch_size": 64,
                "gradient_accumulation_steps": None,  # Will use best from previous phases
                "eval_train_set_frequency": 5,
                "epochs": 50,
                "sequence_params": {
                    "short_seq_len": 40,
                    "long_seq_len": 120,
                    "sample_every": 10
                }
            },
            {
                "description": "Run 3B: Denser Sampling",
                "hidden_dim": None,  # Will use best from previous phases
                "dropout": 0.2,
                "learning_rate": None,  # Will use best from previous phases
                "batch_size": 64,
                "gradient_accumulation_steps": None,  # Will use best from previous phases
                "eval_train_set_frequency": 5,
                "epochs": 50,
                "sequence_params": {
                    "short_seq_len": None,  # Will use best from previous phases
                    "long_seq_len": None,  # Will use best from previous phases
                    "sample_every": 5  # More frequent sampling
                }
            }
        ],
        "PHASE4_LEARNING": [
            {
                "description": "Run 4A: Higher Learning Rate",
                "hidden_dim": None,  # Will use best from previous phases
                "dropout": 0.2,
                "learning_rate": 5e-4,
                "batch_size": 64,
                "gradient_accumulation_steps": None,  # Will use best from previous phases
                "eval_train_set_frequency": 5,
                "epochs": 50,
                "sequence_params": {
                    "short_seq_len": None,  # Will use best from previous phases
                    "long_seq_len": None,  # Will use best from previous phases
                    "sample_every": None  # Will use best from previous phases
                }
            },
            {
                "description": "Run 4B: Lower Learning Rate",
                "hidden_dim": None,  # Will use best from previous phases
                "dropout": 0.2,
                "learning_rate": 5e-5,
                "batch_size": 64,
                "gradient_accumulation_steps": None,  # Will use best from previous phases
                "eval_train_set_frequency": 5,
                "epochs": 50,
                "sequence_params": {
                    "short_seq_len": None,  # Will use best from previous phases
                    "long_seq_len": None,  # Will use best from previous phases
                    "sample_every": None  # Will use best from previous phases
                }
            }
        ],
        "PHASE5_REGULARIZATION": [
            {
                "description": "Run 5A: Higher Dropout",
                "hidden_dim": None,  # Will use best from previous phases
                "dropout": 0.3,
                "learning_rate": None,  # Will use best from previous phases
                "batch_size": 64,
                "gradient_accumulation_steps": None,  # Will use best from previous phases
                "eval_train_set_frequency": 5,
                "epochs": 50,
                "sequence_params": {
                    "short_seq_len": None,  # Will use best from previous phases
                    "long_seq_len": None,  # Will use best from previous phases
                    "sample_every": None  # Will use best from previous phases
                }
            },
            {
                "description": "Run 5B: Lower Dropout",
                "hidden_dim": None,  # Will use best from previous phases
                "dropout": 0.1,
                "learning_rate": None,  # Will use best from previous phases
                "batch_size": 64,
                "gradient_accumulation_steps": None,  # Will use best from previous phases
                "eval_train_set_frequency": 5,
                "epochs": 50,
                "sequence_params": {
                    "short_seq_len": None,  # Will use best from previous phases
                    "long_seq_len": None,  # Will use best from previous phases
                    "sample_every": None  # Will use best from previous phases
                }
            }
        ],
        "PHASE6_FINAL": [
            {
                "description": "Run 6: Final Best Config",
                "hidden_dim": None,  # Will use best from previous phases
                "dropout": None,  # Will use best from previous phases
                "learning_rate": None,  # Will use best from previous phases
                "batch_size": 64,
                "gradient_accumulation_steps": None,  # Will use best from previous phases
                "eval_train_set_frequency": 5,
                "epochs": 100,  # Train longer for final model
                "sequence_params": {
                    "short_seq_len": None,  # Will use best from previous phases
                    "long_seq_len": None,  # Will use best from previous phases
                    "sample_every": None  # Will use best from previous phases
                }
            }
        ]
    },

    # The remaining model types remain unchanged.
    "MLP": {
        "PHASE1_BASELINE": [
            {
                "description": "Run 1: MLP Baseline",
                "hidden_dims": [64, 32],
                "dropout": 0.1,
                "learning_rate": 1e-3,
                "batch_size": 2048,
                "eval_train_set_frequency": 5
            }
        ],
        "PHASE2_ARCHITECTURE": [
            {
                "description": "Run 2A: Wider Network",
                "hidden_dims": [128, 64],
                "dropout": 0.1,
                "learning_rate": 1e-3,
                "batch_size": 2048,
                "eval_train_set_frequency": 5
            },
            {
                "description": "Run 2B: Deeper Network",
                "hidden_dims": [64, 32, 16],
                "dropout": 0.1,
                "learning_rate": 1e-3,
                "batch_size": 2048,
                "eval_train_set_frequency": 5
            }
        ],
        "PHASE3_LEARNING": [
            {
                "description": "Run 3A: Faster Learning",
                "hidden_dims": [64, 32],
                "dropout": 0.1,
                "learning_rate": 3e-3,
                "batch_size": 2048,
                "eval_train_set_frequency": 5
            },
            {
                "description": "Run 3B: Slower Learning",
                "hidden_dims": [64, 32],
                "dropout": 0.1,
                "learning_rate": 3e-4,
                "batch_size": 2048,
                "eval_train_set_frequency": 5
            }
        ],
        "PHASE4_REGULARIZATION": [
            {
                "description": "Run 4A: Higher Dropout",
                "hidden_dims": [64, 32],
                "dropout": 0.2,
                "learning_rate": 1e-3,
                "batch_size": 2048,
                "eval_train_set_frequency": 5
            },
            {
                "description": "Run 4B: Lower Dropout",
                "dropout": 0.05,
                "hidden_dims": [64, 32],
                "learning_rate": 1e-3,
                "batch_size": 2048,
                "eval_train_set_frequency": 5
            }
        ],
        "PHASE5_FINAL": [
            {
                "description": "Run 5: Final Best Config",
                "hidden_dims": None,
                "dropout": None,
                "learning_rate": None,
                "batch_size": 2048,
                "eval_train_set_frequency": 5
            }
        ]
    },

    "LASSO": {
        "PHASE1_BASELINE": [
            {
                "description": "Run 1: Lasso Baseline",
                "alpha": 1.0,
                "max_iter": 1000,
                "tol": 1e-4
            }
        ],
        "PHASE2_ALPHA": [
            {
                "description": "Run 2A: Lower Alpha",
                "alpha": 0.1,
                "max_iter": 1000,
                "tol": 1e-4
            },
            {
                "description": "Run 2B: Higher Alpha",
                "alpha": 10.0,
                "max_iter": 1000,
                "tol": 1e-4
            }
        ],
        "PHASE3_TOLERANCE": [
            {
                "description": "Run 3A: Finer Tolerance",
                "alpha": None,
                "max_iter": 1000,
                "tol": 1e-5
            },
            {
                "description": "Run 3B: Coarser Tolerance",
                "alpha": None,
                "max_iter": 1000,
                "tol": 1e-3
            }
        ],
        "PHASE4_ITERATIONS": [
            {
                "description": "Run 4A: More Iterations",
                "alpha": None,
                "max_iter": 2000,
                "tol": None
            },
            {
                "description": "Run 4B: Fewer Iterations",
                "alpha": None,
                "max_iter": 500,
                "tol": None
            }
        ],
        "PHASE5_FINAL": [
            {
                "description": "Run 5: Final Best Config",
                "alpha": None,
                "max_iter": None,
                "tol": None
            }
        ]
    }
}
