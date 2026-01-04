#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to validate all inverse optimization modules can be imported
"""

import sys

def test_imports():
    """Test importing all optimization modules"""

    print("="*80)
    print("Testing Inverse Optimization Module Imports")
    print("="*80)

    errors = []

    # Test 1: Config module
    print("\n[1/6] Testing optimization.config...")
    try:
        from optimization.config import (
            OptimizationConfig,
            MultiObjectiveConfig,
            KalmanConfig
        )
        print("  ✅ optimization.config imported successfully")
    except Exception as e:
        errors.append(f"config: {str(e)}")
        print(f"  ❌ Failed: {str(e)}")

    # Test 2: Constraints module
    print("\n[2/6] Testing optimization.constraints...")
    try:
        from optimization.constraints import (
            InputConstraint,
            ConstraintManager
        )
        print("  ✅ optimization.constraints imported successfully")
    except Exception as e:
        errors.append(f"constraints: {str(e)}")
        print(f"  ❌ Failed: {str(e)}")

    # Test 3: Inverse optimizer
    print("\n[3/6] Testing optimization.inverse_optimizer...")
    try:
        from optimization.inverse_optimizer import InverseOptimizer
        print("  ✅ optimization.inverse_optimizer imported successfully")
    except Exception as e:
        errors.append(f"inverse_optimizer: {str(e)}")
        print(f"  ❌ Failed: {str(e)}")

    # Test 4: Multi-objective
    print("\n[4/6] Testing optimization.multi_objective...")
    try:
        from optimization.multi_objective import MultiObjectiveOptimizer
        print("  ✅ optimization.multi_objective imported successfully")
    except Exception as e:
        errors.append(f"multi_objective: {str(e)}")
        print(f"  ❌ Failed: {str(e)}")

    # Test 5: Kalman filter
    print("\n[5/6] Testing optimization.kalman_filter...")
    try:
        from optimization.kalman_filter import KalmanCorrector
        print("  ✅ optimization.kalman_filter imported successfully")

        # Check if filterpy is available
        try:
            import filterpy
            print("      ℹ️  filterpy is installed")
        except ImportError:
            print("      ⚠️  filterpy not installed (required for Kalman filter)")
            print("         Install with: pip install filterpy")
    except Exception as e:
        errors.append(f"kalman_filter: {str(e)}")
        print(f"  ❌ Failed: {str(e)}")

    # Test 6: Main package import
    print("\n[6/6] Testing main optimization package...")
    try:
        from optimization import (
            InverseOptimizer,
            MultiObjectiveOptimizer,
            KalmanCorrector,
            ConstraintManager,
            InputConstraint,
            OptimizationConfig
        )
        print("  ✅ optimization package imported successfully")
    except Exception as e:
        errors.append(f"main package: {str(e)}")
        print(f"  ❌ Failed: {str(e)}")

    # Test optional dependencies
    print("\n" + "="*80)
    print("Checking Optional Dependencies")
    print("="*80)

    # Plotly
    print("\n[Optional] plotly (for interactive visualization)...")
    try:
        import plotly
        print(f"  ✅ plotly {plotly.__version__} installed")
    except ImportError:
        print("  ⚠️  plotly not installed")
        print("      Install with: pip install plotly")

    # Filterpy
    print("\n[Optional] filterpy (for Kalman filter)...")
    try:
        import filterpy
        print(f"  ✅ filterpy installed")
    except ImportError:
        print("  ⚠️  filterpy not installed")
        print("      Install with: pip install filterpy")

    # Core dependencies
    print("\n" + "="*80)
    print("Checking Core Dependencies")
    print("="*80)

    deps = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'sklearn': 'Scikit-learn',
        'matplotlib': 'Matplotlib',
        'gradio': 'Gradio'
    }

    for module, name in deps.items():
        try:
            mod = __import__(module)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  ✅ {name}: {version}")
        except ImportError:
            print(f"  ❌ {name}: NOT INSTALLED")
            errors.append(f"Missing dependency: {name}")

    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)

    if not errors:
        print("\n✅ All tests passed successfully!")
        print("\n🚀 You can now use the inverse optimization system:")
        print("   - Run demo: python quick_start_inverse_opt.py")
        print("   - Launch UI: python gradio_apps/inverse_control_interface.py")
        print("   - See docs: cat optimization/README.md")
        return 0
    else:
        print(f"\n❌ {len(errors)} error(s) found:")
        for i, error in enumerate(errors, 1):
            print(f"   {i}. {error}")
        print("\n📝 Please install missing dependencies:")
        print("   pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(test_imports())
