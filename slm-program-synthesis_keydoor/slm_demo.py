"""
Simple Demo Script for KeyDoorSLMEnv and SLMKeyDoorAgent

This script provides a quick demonstration of the new SLM-specific environment and agent.
"""

import logging
from keydoor_slm_env import KeyDoorSLMEnv
from slm_agent import SLMKeyDoorAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def quick_demo():
    """Run a quick demonstration of the SLM system."""
    print("🚀 KeyDoorSLMEnv and SLMKeyDoorAgent - Quick Demo")
    print("=" * 60)
    
    # Initialize environment and agent
    print("🔧 Initializing environment and agent with Llama 3.2 3B...")
    env = KeyDoorSLMEnv(grid_size=8, max_steps=200)
    agent = SLMKeyDoorAgent(model_name="llama3.2:3b", temperature=0.1)
    
    # Test template 1
    print("\n📋 Testing Template 1...")
    obs, info = env.reset(template_id=1)
    agent.reset_stats()
    
    template_info = {
        "template_id": 1,
        "grid_size": env.current_template['grid_size'],
        "keys": env.current_template['keys_pos'],
        "door": env.current_template['door_pos'],
        "agent_start": env.current_template['agent_start']
    }
    
    print(f"   - Template: {template_info['template_id']}")
    print(f"   - Size: {template_info['grid_size']}")
    print(f"   - Keys: {template_info['keys']}")
    print(f"   - Door: {template_info['door']}")
    print(f"   - Agent Start: {template_info['agent_start']}")
    
    # Show spatial blueprint
    print(f"\n🗺️  Spatial Blueprint:")
    blueprint = env.get_spatial_blueprint()
    print(blueprint)
    
    # Run episode
    print(f"\n🎮 Running episode...")
    step_count = 0
    max_steps = 50  # Limit for demo
    
    while step_count < max_steps:
        action = agent.act(env)
        obs, reward, done, info = env.step(action)
        
        current_task = env.get_current_navigation_task()
        print(f"   Step {step_count + 1}: {['UP', 'DOWN', 'LEFT', 'RIGHT', 'INTERACT', 'NO_OP'][action]} "
              f"-> {tuple(env.agent_pos)} (Keys: {env.keys_collected}/{env.total_keys})")
        
        step_count += 1
        
        if done:
            break
    
    # Results
    success = env._are_all_tasks_completed()
    print(f"\n📊 Results:")
    print(f"   - Success: {'✅' if success else '❌'}")
    print(f"   - Steps: {step_count}")
    print(f"   - Keys Collected: {env.keys_collected}/{env.total_keys}")
    print(f"   - Door Status: {'OPEN' if env.door_open else 'CLOSED'}")
    
    # Agent performance
    agent_stats = agent.get_performance_stats()
    print(f"\n🤖 Agent Performance:")
    print(f"   - Requests: {agent_stats['total_requests']}")
    print(f"   - Success Rate: {agent_stats['success_rate']:.1%}")
    print(f"   - Avg Response Time: {agent_stats['avg_response_time']:.2f}s")
    
    print(f"\n✅ Demo completed!")

if __name__ == "__main__":
    quick_demo()
