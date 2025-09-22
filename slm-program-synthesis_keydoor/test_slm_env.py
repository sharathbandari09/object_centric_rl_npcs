"""
Test Script for KeyDoorSLMEnv and SLMKeyDoorAgent

This script demonstrates the new SLM-specific environment and agent.
"""

import logging
import time
from typing import Dict, Any, List

from keydoor_slm_env import KeyDoorSLMEnv
from slm_agent import SLMKeyDoorAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_single_template(env: KeyDoorSLMEnv, agent: SLMKeyDoorAgent, template_id: int) -> Dict[str, Any]:
    """Test a single template with the SLM agent."""
    print(f"\n{'='*60}")
    print(f"TESTING TEMPLATE {template_id} WITH SLM AGENT")
    print(f"{'='*60}")
    
    # Reset environment
    obs, info = env.reset(template_id=template_id)
    agent.reset_stats()
    
    # Get initial state
    template_info = {
        "template_id": template_id,
        "grid_size": env.current_template['grid_size'],
        "keys": env.current_template['keys_pos'],
        "door": env.current_template['door_pos'],
        "agent_start": env.current_template['agent_start']
    }
    
    print(f"📋 Template Info:")
    print(f"   - ID: {template_info['template_id']}")
    print(f"   - Size: {template_info['grid_size']}")
    print(f"   - Keys: {template_info['keys']}")
    print(f"   - Door: {template_info['door']}")
    print(f"   - Agent Start: {template_info['agent_start']}")
    
    # Run episode
    print(f"\n🚀 Starting episode...")
    start_time = time.time()
    
    episode_log = []
    step_count = 0
    max_steps = 200
    
    while step_count < max_steps:
        # Get action from agent
        action = agent.act(env)
        
        # Execute action
        obs, reward, done, info = env.step(action)
        
        # Log step
        current_task = env.get_current_navigation_task()
        step_log = {
            "step": step_count + 1,
            "action": action,
            "action_name": ["UP", "DOWN", "LEFT", "RIGHT", "INTERACT", "NO_OP"][action],
            "reward": reward,
            "done": done,
            "agent_pos": tuple(env.agent_pos),
            "keys_collected": env.keys_collected,
            "total_keys": env.total_keys,
            "door_open": env.door_open,
            "current_task": current_task['task_id'] if current_task else None,
            "remaining_tasks": len(env.navigation_tasks) - env.current_task_index
        }
        episode_log.append(step_log)
        
        # Print progress every 10 steps
        if step_count % 10 == 0:
            print(f"   Step {step_count + 1}: {step_log['action_name']} -> {step_log['agent_pos']} "
                  f"(Keys: {step_log['keys_collected']}/{step_log['total_keys']}, "
                  f"Door: {'OPEN' if step_log['door_open'] else 'CLOSED'})")
        
        step_count += 1
        
        if done:
            break
    
    episode_time = time.time() - start_time
    
    # Calculate results
    success = env._are_all_tasks_completed()
    tasks_completed = len(env.completed_tasks)
    total_tasks = len(env.navigation_tasks)
    
    print(f"\n📊 Episode Results:")
    print(f"   - Success: {'✅' if success else '❌'}")
    print(f"   - Steps: {step_count}")
    print(f"   - Tasks Completed: {tasks_completed}/{total_tasks}")
    print(f"   - Episode Time: {episode_time:.2f}s")
    print(f"   - Final Position: {tuple(env.agent_pos)}")
    print(f"   - Keys Collected: {env.keys_collected}/{env.total_keys}")
    print(f"   - Door Status: {'OPEN' if env.door_open else 'CLOSED'}")
    
    # Show completed tasks
    if env.completed_tasks:
        print(f"\n✅ Completed Tasks:")
        for task in env.completed_tasks:
            print(f"   - {task['task_id']}: {task['task_type']} at {task['target_position']}")
    
    # Show remaining tasks
    if env.current_task_index < len(env.navigation_tasks):
        print(f"\n⏳ Remaining Tasks:")
        for i in range(env.current_task_index, len(env.navigation_tasks)):
            task = env.navigation_tasks[i]
            print(f"   - {task['task_id']}: {task['task_type']} at {task['target_position']}")
    
    # Agent performance stats
    agent_stats = agent.get_performance_stats()
    print(f"\n🤖 Agent Performance:")
    print(f"   - Total Requests: {agent_stats['total_requests']}")
    print(f"   - Success Rate: {agent_stats['success_rate']:.1%}")
    print(f"   - Avg Response Time: {agent_stats['avg_response_time']:.2f}s")
    
    return {
        "template_id": template_id,
        "success": success,
        "steps_taken": step_count,
        "tasks_completed": tasks_completed,
        "total_tasks": total_tasks,
        "episode_time": episode_time,
        "episode_log": episode_log,
        "agent_stats": agent_stats
    }

def test_multiple_templates(env: KeyDoorSLMEnv, agent: SLMKeyDoorAgent, template_ids: List[int]) -> Dict[str, Any]:
    """Test multiple templates with the SLM agent."""
    print(f"\n{'='*80}")
    print(f"TESTING MULTIPLE TEMPLATES: {template_ids}")
    print(f"{'='*80}")
    
    all_results = []
    template_results = {}
    
    for template_id in template_ids:
        try:
            results = test_single_template(env, agent, template_id)
            all_results.append(results)
            template_results[template_id] = results
        except Exception as e:
            print(f"❌ Error testing template {template_id}: {e}")
            continue
    
    # Calculate overall statistics
    total_episodes = len(all_results)
    successful_episodes = sum(1 for r in all_results if r['success'])
    avg_steps = sum(r['steps_taken'] for r in all_results) / total_episodes if total_episodes > 0 else 0
    avg_time = sum(r['episode_time'] for r in all_results) / total_episodes if total_episodes > 0 else 0
    
    print(f"\n📊 Overall Results:")
    print(f"   - Total Episodes: {total_episodes}")
    print(f"   - Successful Episodes: {successful_episodes}")
    print(f"   - Success Rate: {successful_episodes / total_episodes:.1%}" if total_episodes > 0 else "   - Success Rate: 0%")
    print(f"   - Avg Steps: {avg_steps:.1f}")
    print(f"   - Avg Time: {avg_time:.2f}s")
    
    # Template-specific results
    print(f"\n📋 Template-Specific Results:")
    for template_id, results in template_results.items():
        print(f"   - Template {template_id}: {'✅' if results['success'] else '❌'} "
              f"({results['steps_taken']} steps, {results['episode_time']:.2f}s)")
    
    return {
        "total_episodes": total_episodes,
        "successful_episodes": successful_episodes,
        "overall_success_rate": successful_episodes / total_episodes if total_episodes > 0 else 0,
        "avg_steps_per_episode": avg_steps,
        "avg_time_per_episode": avg_time,
        "template_results": template_results
    }

def test_spatial_blueprint(env: KeyDoorSLMEnv, template_id: int):
    """Test the spatial blueprint generation."""
    print(f"\n{'='*60}")
    print(f"TESTING SPATIAL BLUEPRINT - TEMPLATE {template_id}")
    print(f"{'='*60}")
    
    # Reset environment
    obs, info = env.reset(template_id=template_id)
    
    # Get spatial blueprint
    blueprint = env.get_spatial_blueprint()
    print(f"📋 Spatial Blueprint:")
    print(blueprint)
    
    # Get navigation context
    context = env.get_navigation_context()
    print(f"\n🧭 Navigation Context:")
    print(f"   - Agent Position: {context['agent_position']}")
    print(f"   - Current Task: {context['current_task']}")
    print(f"   - Target Position: {context['target_position']}")
    print(f"   - Target ID: {context['target_id']}")
    print(f"   - Keys Collected: {context['keys_collected']}/{context['total_keys']}")
    print(f"   - Door Open: {context['door_open']}")
    print(f"   - Remaining Tasks: {context['remaining_tasks']}")

def main():
    """Main test function."""
    print("🚀 KeyDoorSLMEnv and SLMKeyDoorAgent Test")
    print("=" * 60)
    
    # Initialize environment and agent
    print("🔧 Initializing environment and agent...")
    env = KeyDoorSLMEnv(grid_size=8, max_steps=200)
    agent = SLMKeyDoorAgent(model_name="llama3.2:3b", temperature=0.1)
    
    # Test 1: Spatial blueprint generation
    print("\n" + "="*80)
    print("TEST 1: SPATIAL BLUEPRINT GENERATION")
    print("="*80)
    test_spatial_blueprint(env, 1)
    
    # Test 2: Single template test
    print("\n" + "="*80)
    print("TEST 2: SINGLE TEMPLATE TEST")
    print("="*80)
    test_single_template(env, agent, 1)
    
    # Test 3: Multiple templates test
    print("\n" + "="*80)
    print("TEST 3: MULTIPLE TEMPLATES TEST")
    print("="*80)
    test_multiple_templates(env, agent, [1, 2, 3])
    
    print(f"\n✅ All tests completed!")

if __name__ == "__main__":
    main()
