# Summary: Basic Ghost Logger usage example
# Comments: Demonstrates simple research documentation workflow
# Expect: Automatic capture of execution details and any generated files
# Notes: Template for new users to get started quickly

from ghost_logger import capture_everything
import numpy as np
import matplotlib.pyplot as plt

@capture_everything
def basic_research_example():
    """Simple example showing Ghost Logger in action."""
    
    # Generate some sample data
    x = np.linspace(0, 10, 100)
    y = np.sin(x) * np.exp(-x/5)
    
    # Create a plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', linewidth=2, label='Damped sine wave')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Basic Research Example')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('example_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Some calculations
    max_value = np.max(y)
    min_value = np.min(y)
    mean_value = np.mean(y)
    
    print(f"Data analysis results:")
    print(f"Maximum value: {max_value:.4f}")
    print(f"Minimum value: {min_value:.4f}")
    print(f"Mean value: {mean_value:.4f}")
    
    # Save results to file
    results = {
        'max': max_value,
        'min': min_value,
        'mean': mean_value,
        'data_points': len(x)
    }
    
    import json
    with open('analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    basic_research_example()
