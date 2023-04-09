import json
import time
from pathlib import Path

from utils.plot_trace import plot_trace
from utils.runners import run_session

run_tournament = False

if not run_tournament:
    RESULTS_DIR = Path("results", time.strftime('%Y%m%d-%H%M%S'))

    # create results directory if it does not exist
    if not RESULTS_DIR.exists():
        RESULTS_DIR.mkdir(parents=True)

    # Settings to run a negotiation session:
    #   You need to specify the classpath of 2 agents to start a negotiation. Parameters for the agent can be added as a dict (see example)
    #   You need to specify the preference profiles for both agents. The first profile will be assigned to the first agent.
    #   You need to specify a time deadline (is milliseconds (ms)) we are allowed to negotiate before we end without agreement
    settings = {
        "agents": [
            {
                "class": "agents.ANL2022.dreamteam109_agent.dreamteam109_agent.DreamTeam109Agent",
                "parameters": {"storage_dir": "agent_storage/DreamTeam109Agent"},
            },
            {
                "class": "agents.agent_group34.agent_group34.AgentGroup34",
                "parameters": {"storage_dir": "agent_storage/AgentGroup34"},
            },
        ],
        "profiles": ["domains/domain00/profileA.json", "domains/domain00/profileB.json"],
        "deadline_time_ms": 10000,
    }

    # run a session and obtain results in dictionaries
    session_results_trace, session_results_summary = run_session(settings)

    # plot trace to html file
    if not session_results_trace["error"]:
        plot_trace(session_results_trace, RESULTS_DIR.joinpath("trace_plot.html"))

    # write results to file
    with open(RESULTS_DIR.joinpath("session_results_trace.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(session_results_trace, indent=2))
    with open(RESULTS_DIR.joinpath("session_results_summary.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(session_results_summary, indent=2))

else:

    agents = [
        {
            "class": "agents.boulware_agent.boulware_agent.BoulwareAgent",
        },
        {
            "class": "agents.conceder_agent.conceder_agent.ConcederAgent",
        },
        {
            "class": "agents.hardliner_agent.hardliner_agent.HardlinerAgent",
        },
        {
            "class": "agents.linear_agent.linear_agent.LinearAgent",
        },
        {
            "class": "agents.random_agent.random_agent.RandomAgent",
        },
        {
            "class": "agents.stupid_agent.stupid_agent.StupidAgent",
        },
        {
            "class": "agents.CSE3210.agent2.agent2.Agent2",
        },
        {
            "class": "agents.CSE3210.agent3.agent3.Agent3",
        },
        {
            "class": "agents.CSE3210.agent7.agent7.Agent7",
        },
        {
            "class": "agents.CSE3210.agent11.agent11.Agent11",
        },
        {
            "class": "agents.CSE3210.agent14.agent14.Agent14",
        },
        {
            "class": "agents.CSE3210.agent18.agent18.Agent18",
        },
        {
            "class": "agents.CSE3210.agent19.agent19.Agent19",
        },
        {
            "class": "agents.CSE3210.agent22.agent22.Agent22",
        },
        {
            "class": "agents.CSE3210.agent24.agent24.Agent24",
        },
        {
            "class": "agents.CSE3210.agent25.agent25.Agent25",
        },
        {
            "class": "agents.CSE3210.agent26.agent26.Agent26",
        },
        {
            "class": "agents.CSE3210.agent27.agent27.Agent27",
        },
        {
            "class": "agents.CSE3210.agent29.agent29.Agent29",
        },
        {
            "class": "agents.CSE3210.agent32.agent32.Agent32",
        },
        {
            "class": "agents.CSE3210.agent33.agent33.Agent33",
        },
        {
            "class": "agents.CSE3210.agent41.agent41.Agent41",
        },
        {
            "class": "agents.CSE3210.agent43.agent43.Agent43",
        },
        {
            "class": "agents.CSE3210.agent50.agent50.Agent50",
        },
        {
            "class": "agents.CSE3210.agent52.agent52.Agent52",
        },
        {
            "class": "agents.CSE3210.agent55.agent55.Agent55",
        },
        {
            "class": "agents.CSE3210.agent58.agent58.Agent58",
        },
        {
            "class": "agents.CSE3210.agent61.agent61.Agent61",
        },
        {
            "class": "agents.CSE3210.agent64.agent64.Agent64",
        },
        {
            "class": "agents.CSE3210.agent67.agent67.Agent67",
        },
        {
            "class": "agents.CSE3210.agent68.agent68.Agent68",
        },
    ]

    template_agent = {
        "class": "agents.agent_group34.agent_group34.AgentGroup34",
        "parameters": {"storage_dir": "agent_storage/AgentGroup34"},
    }

    for agent in agents:

        RESULTS_DIR = Path("results", time.strftime('%Y%m%d-%H%M%S'))

        # create results directory if it does not exist
        if not RESULTS_DIR.exists():
            RESULTS_DIR.mkdir(parents=True)

        settings = {
            "agents": [agent, template_agent],
            "profiles": ["domains/domain00/profileA.json", "domains/domain00/profileB.json"],
            "deadline_time_ms": 10000,
        }

        # run a session and obtain results in dictionaries
        session_results_trace, session_results_summary = run_session(settings)

        # plot trace to html file
        if not session_results_trace["error"]:
            plot_trace(session_results_trace, RESULTS_DIR.joinpath("trace_plot.html"))

        # write results to file
        with open(RESULTS_DIR.joinpath("session_results_trace.json"), "w", encoding="utf-8") as f:
            f.write(json.dumps(session_results_trace, indent=2))
        with open(RESULTS_DIR.joinpath("session_results_summary.json"), "w", encoding="utf-8") as f:
            f.write(json.dumps(session_results_summary, indent=2))
