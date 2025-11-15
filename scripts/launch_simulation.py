"""
Interactive start menu for configuring and launching evacuation simulations.

Features:
    • Scenario browser with curated layouts.
    • Parameter editing (responder counts, hazard scaling, evacuee density, runtime).
    • Launch options for headless analytics or live visualization.
"""

from __future__ import annotations

import copy
import os
import sys
import tempfile
import textwrap
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from notebooks import build_world, simulate
from haso_sim.animate_live import create_live_visualization
from haso_sim.utils import generate_summary_report


BASE_DIR = Path(__file__).parent
MAP_DIR = BASE_DIR / "notebooks" / "data" / "scenarios"
CONFIG_DIR = BASE_DIR / "notebooks" / "data" / "configs"

SCENARIOS = [
    {
        "key": "innovation_hub",
        "map": MAP_DIR / "innovation_hub.yaml",
        "config": CONFIG_DIR / "innovation_hub_config.yaml",
    },
    {
        "key": "medical_pavilion",
        "map": MAP_DIR / "medical_pavilion.yaml",
        "config": CONFIG_DIR / "medical_pavilion_config.yaml",
    },
    {
        "key": "transit_atrium",
        "map": MAP_DIR / "transit_atrium.yaml",
        "config": CONFIG_DIR / "transit_atrium_config.yaml",
    },
]


def print_banner() -> None:
    banner = r"""
 __  __  _____  __  __   _____  __  __   _____   _____   __  __
|  \/  || ____||  \/  | |_   _||  \/  | | ____| |_   _| |  \/  |
| |\/| ||  _|  | |\/| |   | |  | |\/| | |  _|     | |   | |\/| |
| |  | || |___ | |  | |   | |  | |  | | | |___    | |   | |  | |
|_|  |_||_____||_|  |_|   |_|  |_|  |_| |_____|   |_|   |_|  |_|
"""
    print(banner)
    print("HiMCM Evacuation Simulation Launcher")
    print("====================================\n")


def safe_load_yaml(path: Path) -> Dict[str, Any]:
    if not path or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def estimate_evacuees(map_data: Dict[str, Any]) -> int:
    count = len(map_data.get("evacuees", []))
    for group in map_data.get("evacuee_groups", []):
        count += int(group.get("count", 0) or 0)
    return count


def load_scenario_metadata(entry: Dict[str, Any]) -> Dict[str, Any]:
    map_data = safe_load_yaml(Path(entry["map"]))
    config_data = safe_load_yaml(Path(entry["config"]))

    title = config_data.get("title", entry["key"].replace("_", " ").title())
    summary = config_data.get("summary", "No summary provided.")
    evac_count = estimate_evacuees(map_data)

    recommended = config_data.get("menu", {}).get("recommended_roles", {})
    responder_count = sum(int(info.get("count", 0)) for info in recommended.values()) or len(
        config_data.get("agents", [])
    )

    difficulty = config_data.get("menu", {}).get("difficulty", "Scenario")

    return {
        "title": title,
        "summary": summary,
        "evacuees": evac_count,
        "responders": responder_count,
        "difficulty": difficulty,
        "map_data": map_data,
        "config_data": config_data,
    }


def prompt_choice(prompt: str, options: List[str]) -> str:
    choice = input(prompt).strip().lower()
    while choice not in options:
        choice = input(f"Please choose {', '.join(options)}: ").strip().lower()
    return choice


def prompt_int(prompt: str, default: int) -> int:
    raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return default
    try:
        value = int(raw)
        return max(0, value)
    except ValueError:
        print("Enter a valid integer.")
        return prompt_int(prompt, default)


def prompt_float(prompt: str, default: float, minimum: float = 0.0, maximum: float = None) -> float:
    raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        print("Enter a valid number.")
        return prompt_float(prompt, default, minimum, maximum)
    if value < minimum:
        print(f"Value must be ≥ {minimum}.")
        return prompt_float(prompt, default, minimum, maximum)
    if maximum is not None and value > maximum:
        print(f"Value must be ≤ {maximum}.")
        return prompt_float(prompt, default, minimum, maximum)
    return value


def build_role_templates(config: Dict[str, Any]) -> Tuple[OrderedDict, Dict[str, int]]:
    templates: OrderedDict[str, Dict[str, Any]] = OrderedDict()
    default_counts: Dict[str, int] = {}

    menu_roles = config.get("menu", {}).get("recommended_roles", {})
    if menu_roles:
        for role, info in menu_roles.items():
            role_name = role.upper()
            templates[role_name] = {
                "role": role_name,
                "node": info.get("node", 0),
                "sweep_mode": info.get("sweep_mode", "right"),
                "personal_priority": info.get("personal_priority", 3),
            }
            default_counts[role_name] = int(info.get("count", 0) or 0)
    else:
        for agent in config.get("agents", []):
            role_name = agent.get("role", "SCOUT").upper()
            if role_name not in templates:
                templates[role_name] = {
                    "role": role_name,
                    "node": agent.get("node", 0),
                    "sweep_mode": agent.get("sweep_mode", "right"),
                    "personal_priority": agent.get("personal_priority", 3),
                }
            default_counts[role_name] = default_counts.get(role_name, 0) + 1

    return templates, default_counts


def apply_role_counts(config: Dict[str, Any], templates: OrderedDict, counts: Dict[str, int]) -> None:
    new_agents = []
    next_id = 0
    for role, template in templates.items():
        count = counts.get(role, 0)
        for _ in range(count):
            agent_cfg = {
                "id": next_id,
                "role": role,
                "node": template.get("node", 0),
                "sweep_mode": template.get("sweep_mode", "right"),
                "personal_priority": template.get("personal_priority", 3),
            }
            new_agents.append(agent_cfg)
            next_id += 1
    config["agents"] = new_agents


def scale_hazards(map_data: Dict[str, Any], scale: float) -> Dict[str, Any]:
    if abs(scale - 1.0) < 1e-3:
        return map_data
    adjusted = copy.deepcopy(map_data)
    for node in adjusted.get("nodes", []):
        hazard = node.get("hazard", "NONE").upper()
        if hazard != "NONE":
            severity = float(node.get("hazard_severity", 0.0))
            severity = max(0.0, min(1.0, round(severity * scale, 3)))
            node["hazard_severity"] = severity
            if severity <= 0.0:
                node["hazard"] = "NONE"
    return adjusted


def scale_evacuees(map_data: Dict[str, Any], scale: float) -> Dict[str, Any]:
    if abs(scale - 1.0) < 1e-3:
        return map_data
    adjusted = copy.deepcopy(map_data)
    groups = []
    for group in adjusted.get("evacuee_groups", []):
        base = int(group.get("count", 0) or 0)
        new_count = int(round(base * scale))
        if new_count <= 0:
            continue
        group["count"] = new_count
        groups.append(group)
    adjusted["evacuee_groups"] = groups
    return adjusted


def write_temp_yaml(data: Dict[str, Any]) -> str:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8")
    yaml.safe_dump(data, tmp, sort_keys=False)
    tmp.flush()
    tmp.close()
    return tmp.name


def display_parameters(state: Dict[str, Any]) -> None:
    print("\nCurrent Configuration")
    print("---------------------")
    print(f"Scenario:          {state['scenario_name']}")
    print(f"Mode:              {state['mode'].title()}")
    print(f"Duration (s):      {state['duration']}")
    print(f"Visualization FPS: {state['fps']}")
    print(f"Random Seed:       {state['seed']}")
    print(f"Hazard Scale:      {state['hazard_scale']:.2f}")
    print(f"Evacuee Scale:     {state['evacuee_scale']:.2f}")
    print("Responder Counts:")
    for role, count in state["role_counts"].items():
        print(f"  - {role.title():<12} {count}")
    print("")


def parameter_menu(state: Dict[str, Any]) -> None:
    templates = state["role_templates"]
    while True:
        display_parameters(state)
        print("Parameter Menu")
        print(" 1) Adjust responder counts")
        print(" 2) Scale hazard severity")
        print(" 3) Scale evacuee density")
        print(" 4) Set duration (seconds)")
        print(" 5) Set visualization FPS")
        print(" 6) Set random seed")
        print(" 7) Toggle mode (headless / interactive)")
        print(" 8) Launch simulation")
        print(" 9) Quit\n")

        choice = prompt_choice("Select an option: ", list("123456789"))

        if choice == "1":
            for role in templates.keys():
                default = state["role_counts"].get(role, 0)
                state["role_counts"][role] = prompt_int(f"  {role.title()} count", default)
        elif choice == "2":
            state["hazard_scale"] = prompt_float("Hazard severity scale", state["hazard_scale"], minimum=0.0)
        elif choice == "3":
            state["evacuee_scale"] = prompt_float("Evacuee density scale", state["evacuee_scale"], minimum=0.1)
        elif choice == "4":
            state["duration"] = prompt_int("Simulation duration (seconds)", state["duration"])
        elif choice == "5":
            state["fps"] = prompt_int("Visualization FPS", state["fps"])
        elif choice == "6":
            state["seed"] = prompt_int("Random seed", state["seed"])
        elif choice == "7":
            state["mode"] = "headless" if state["mode"] == "interactive" else "interactive"
        elif choice == "8":
            launch_simulation(state)
        elif choice == "9":
            print("Goodbye!")
            sys.exit(0)


def launch_simulation(state: Dict[str, Any]) -> None:
    base_map = state["map_data"]
    base_config = state["config_data"]

    working_map = scale_evacuees(scale_hazards(base_map, state["hazard_scale"]), state["evacuee_scale"])
    working_config = copy.deepcopy(base_config)

    apply_role_counts(working_config, state["role_templates"], state["role_counts"])
    working_config.setdefault("simulation", {})
    working_config["simulation"]["tmax"] = state["duration"]
    working_config["simulation"]["seed"] = state["seed"]

    temp_files = []
    try:
        map_path = state["map_path"]
        if working_map is not base_map:
            map_path = write_temp_yaml(working_map)
            temp_files.append(map_path)

        config_path = state["config_path"]
        if working_config is not base_config:
            config_path = write_temp_yaml(working_config)
            temp_files.append(config_path)

        if state["mode"] == "interactive":
            print("\nLaunching live dashboard... close the window to return to the menu.")
            world = build_world(map_path, config_path=config_path, seed=state["seed"])
            create_live_visualization(
                world,
                fps=state["fps"],
                duration=state["duration"],
                save_video=False,
                advanced=True,
                quiet=False,
            )
        else:
            print("\nRunning headless simulation...")
            result = simulate(
                map_path,
                config_path=config_path,
                tmax=state["duration"],
                seed=state["seed"],
                animate=False,
            )
            world = result["world"]
            print()
            print(generate_summary_report(world))
            print()
    finally:
        for file_path in temp_files:
            try:
                os.unlink(file_path)
            except OSError:
                pass


def choose_scenario() -> Dict[str, Any]:
    print("Scenario Browser")
    print("----------------")
    for idx, entry in enumerate(SCENARIOS, start=1):
        metadata = load_scenario_metadata(entry)
        wrapped_summary = textwrap.fill(metadata["summary"], width=70)
        print(f"{idx}) {metadata['title']}")
        print(f"   • Evacuees: {metadata['evacuees']:>3}  • Recommended Responders: {metadata['responders']}")
        print(f"   • Overview: {wrapped_summary}\n")
        entry["metadata"] = metadata

    print("0) Custom scenario (provide your own map/config paths)\n")

    while True:
        raw = input("Select a scenario: ").strip()
        if raw == "0":
            map_path = Path(input(" Map file (.yaml): ").strip())
            config_path_input = input(" Config file (.yaml, optional): ").strip()
            config_path = Path(config_path_input) if config_path_input else None
            map_data = safe_load_yaml(map_path)
            config_data = safe_load_yaml(config_path) if config_path else {}
            title = config_data.get("title", map_path.stem.title())
            summary = config_data.get("summary", "Custom scenario provided by user.")
            metadata = {
                "title": title,
                "summary": summary,
                "evacuees": estimate_evacuees(map_data),
                "responders": len(config_data.get("agents", [])),
                "map_data": map_data,
                "config_data": config_data,
            }
            return {
                "scenario_name": title,
                "map_path": str(map_path),
                "config_path": str(config_path) if config_path else None,
                "metadata": metadata,
            }
        try:
            choice = int(raw)
            if 1 <= choice <= len(SCENARIOS):
                entry = SCENARIOS[choice - 1]
                entry["map_path"] = str(entry["map"])
                entry["config_path"] = str(entry["config"])
                entry["scenario_name"] = entry["metadata"]["title"]
                return entry
        except ValueError:
            pass
        print("Invalid selection. Try again.")


def main() -> None:
    print_banner()
    scenario_entry = choose_scenario()

    metadata = scenario_entry.get("metadata") or load_scenario_metadata(scenario_entry)
    config_data = metadata["config_data"]
    map_data = metadata["map_data"]

    templates, default_counts = build_role_templates(config_data)
    if not templates:
        templates = OrderedDict({
            "SCOUT": {
                "role": "SCOUT",
                "node": 0,
                "sweep_mode": "right",
                "personal_priority": 3,
            }
        })
        default_counts = {"SCOUT": max(1, len(config_data.get("agents", [])) or 1)}

    defaults = config_data.get("menu", {}).get("defaults", {})
    state = {
        "scenario_name": scenario_entry.get("scenario_name", metadata["title"]),
        "map_path": scenario_entry.get("map_path"),
        "config_path": scenario_entry.get("config_path"),
        "map_data": map_data,
        "config_data": config_data,
        "role_templates": templates,
        "role_counts": {role: default_counts.get(role, 0) for role in templates.keys()},
        "hazard_scale": 1.0,
        "evacuee_scale": 1.0,
        "duration": int(defaults.get("duration", config_data.get("simulation", {}).get("tmax", 1200))),
        "fps": int(defaults.get("fps", 20)),
        "seed": int(defaults.get("seed", config_data.get("simulation", {}).get("seed", 42))),
        "mode": "interactive",
    }

    parameter_menu(state)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSimulation launcher interrupted.")

