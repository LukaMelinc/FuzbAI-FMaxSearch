import multiprocessing as mp
import torch
import time
from strel import ContinuousAgent
from FuzbAISim import FuzbAISim


def simulation_worker(instance_id, experience_queue, model_weights, stop_event):
    """
    Worker for running a single simulation instance.
    """
    agent = ContinuousAgent()
    agent.model.load_state_dict(model_weights.get())

    simulator = FuzbAISim()
    simulator.run()

    try:
        while not stop_event.is_set():
            # Get camera data and process it
            cam_data = simulator.getDelayedCamera(1, time.time())
            if cam_data:
                motor_cmds = agent.process_data(cam_data)
                
                # Share experience with the central learner
                for experience in agent.memory.buffer:
                    experience_queue.put(experience)
                agent.memory.buffer.clear()  # so we don't repeat them next time


                # Update agent's model with the latest weights
                if not model_weights.empty():
                    agent.model.load_state_dict(model_weights.get())

            time.sleep(0.02)

    except KeyboardInterrupt:
        print(f"Simulation {instance_id} stopped by user.")
    finally:
        simulator.stop()


def central_learner(num_instances, experience_queue, model_weights, stop_event):
    """
    Central process for training the shared model.
    """
    agent = ContinuousAgent()
    batch_size = agent.batch_size

    try:
        while not stop_event.is_set():
            # Collect experiences from all instances
            experiences = []
            while not experience_queue.empty():
                experiences.append(experience_queue.get())

            # Train if enough experiences are collected
            if len(experiences) >= batch_size:
                batch = experiences[:batch_size]
                states, actions, rewards, next_states, dones = zip(*batch)
                
                agent.learn_step_counter += 1
                agent.learn_from_batch(states, actions, rewards, next_states, dones)
                
                # Share updated weights with all instances
                model_weights.put(agent.model.state_dict())

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Central learning stopped by user.")
    finally:
        agent.save_model("distributed_trained_model.pth")
        stop_event.set()


def start_distributed_training(num_instances):
    """
    Starts distributed training with multiple simulation instances.
    """
    experience_queue = mp.Queue()
    model_weights = mp.Queue()
    stop_event = mp.Event()

    # Initialize and share initial weights
    initial_agent = ContinuousAgent()
    model_weights.put(initial_agent.model.state_dict())

    # Start simulation workers
    workers = []
    for i in range(num_instances):
        p = mp.Process(target=simulation_worker, args=(i, experience_queue, model_weights, stop_event))
        workers.append(p)
        p.start()

    # Start central learner
    learner = mp.Process(target=central_learner, args=(num_instances, experience_queue, model_weights, stop_event))
    learner.start()

    try:
        # Run indefinitely
        learner.join()
    except KeyboardInterrupt:
        print("Stopping distributed training...")
        stop_event.set()
    finally:
        for p in workers:
            p.join()


if __name__ == "__main__":
    start_distributed_training(num_instances=3)
