"""
Simple test for SQL tools system.
"""

def test_imports():
    """Test that all imports work correctly."""
    try:
        from .controller import create_sql_controller
        from .models import SQLGenerationResponse
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_controller_creation():
    """Test that the controller can be created."""
    try:
        from .controller import create_sql_controller
        controller = create_sql_controller()
        print("✓ Controller created successfully")
        return True
    except Exception as e:
        print(f"✗ Controller creation failed: {e}")
        return False

def main():
    """Run simple tests."""
    print("=== Simple SQL Tools Tests ===\n")
    
    tests = [
        test_imports,
        test_controller_creation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed")

if __name__ == "__main__":
    main()