# vim /openhands/python/lib/python3.12/site-packages/openhands/sdk/agent/agent.py # FIXME: 注释掉 277-280
# vim /openhands/python/lib/python3.12/site-packages/openhands/sdk/conversation/impl/local_conversation.py

import json
import logging
import os
import types
from pathlib import Path
from typing import List

from jinja2 import Environment, FileSystemLoader

from benchmarks.utils.args_parser import get_parser
from benchmarks.utils.critics import create_critic
from benchmarks.utils.evaluation import Evaluation
from benchmarks.utils.evaluation_utils import (
    construct_eval_output_dir,
    get_default_on_result_writer,
)
from benchmarks.utils.models import (
    EvalInstance,
    EvalMetadata,
    EvalOutput,
)
from openhands.sdk import LLM, Agent, Conversation, get_logger
from openhands.sdk.workspace import LocalWorkspace, RemoteWorkspace
from openhands.tools.preset.default import get_default_tools

from openhands.sdk.observability.laminar import observe
from openhands.sdk.agent.utils import make_llm_completion, prepare_llm_messages
from openhands.sdk.conversation import (
    ConversationCallbackType,
    ConversationState,
    ConversationTokenCallbackType,
    LocalConversation,
)
from openhands.sdk.event import ActionEvent, MessageEvent
from openhands.sdk.event.condenser import Condensation, CondensationRequest
from openhands.sdk.llm import Message, TextContent
from openhands.sdk.llm.exceptions import (
    FunctionCallValidationError,
    LLMContextWindowExceedError,
)


logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = get_logger(__name__)
logger.setLevel(logging.DEBUG)
for h in logger.handlers:
    h.setLevel(logging.DEBUG)


def make_instance(inst_file: str) -> EvalInstance:
    required_keys = (
        "instance_id",
        "image_name",
        "problem_statement",
        "repo_name",
        "base_commit",
        "remote_user",
        "project_path",
        "script_folder",
        "remote_workspace_folder",
        "FAIL_TO_PASS",
    )

    with open(inst_file, "r") as f:
        curr_instance: dict = json.loads(f.read())
        for key in required_keys:
            if key not in curr_instance:
                logger.warning(f"{key} is missing...")

        instance = EvalInstance(
            id=curr_instance["instance_id"],
            data={
                "repo": curr_instance["repo_name"],
                "project_path": curr_instance["project_path"].rstrip(
                    "/"
                ),  # repo data store path
                "problem_statement": curr_instance["problem_statement"],
                "base_commit": curr_instance.get("base_commit", ""),
                "image_name": curr_instance.get("image_name", ""),
                "FAIL_TO_PASS": curr_instance.get("FAIL_TO_PASS", ""),
                "remote_user": curr_instance.get("remote_user", ""),
                "repo_path": f"/workspace/{curr_instance['repo_name'].split('/')[-1]}",  # Agent work place
                "script_folder": curr_instance.get(
                    "script_folder", ""
                ),  # test script directory
            },
        )

        if not os.path.exists(instance.data["repo_path"]):
            os.makedirs(instance.data["repo_path"])

    return instance

@observe(name="agent.step", ignore_inputs=["state", "on_event"])
def _hijack_step(
    self,
    conversation: LocalConversation,
    on_event: ConversationCallbackType,
    on_token: ConversationTokenCallbackType | None = None,
) -> None:
    state = conversation.state
    # Check for pending actions (implicit confirmation)
    # and execute them before sampling new actions.
    pending_actions = ConversationState.get_unmatched_actions(state.events)
    if pending_actions:
        logger.info(
            "Confirmation mode: Executing %d pending action(s)",
            len(pending_actions),
        )
        self._execute_actions(conversation, pending_actions, on_event)
        return

    # Prepare LLM messages using the utility function
    _messages_or_condensation = prepare_llm_messages(
        state.events, condenser=self.condenser
    )

    # Process condensation event before agent samples another action
    if isinstance(_messages_or_condensation, Condensation):
        on_event(_messages_or_condensation)
        return

    _messages = _messages_or_condensation

    logger.debug(
        "Sending messages to LLM: "
        f"{json.dumps([m.model_dump() for m in _messages[1:]], indent=2)}"
    )

    try:
        llm_response = make_llm_completion(
            self.llm,
            _messages,
            tools=list(self.tools_map.values()),
            on_token=on_token,
        )
    except FunctionCallValidationError as e:
        logger.warning(f"LLM generated malformed function call: {e}")
        error_message = MessageEvent(
            source="user",
            llm_message=Message(
                role="user",
                content=[TextContent(text=str(e))],
            ),
        )
        on_event(error_message)
        return
    except LLMContextWindowExceedError as e:
        # If condenser is available and handles requests, trigger condensation
        if (
            self.condenser is not None
            and self.condenser.handles_condensation_requests()
        ):
            logger.warning(
                "LLM raised context window exceeded error, triggering condensation"
            )
            on_event(CondensationRequest())
            return
        # No condenser available or doesn't handle requests; log helpful warning
        self._log_context_window_exceeded_warning()
        raise e

    # LLMResponse already contains the converted message and metrics snapshot
    message: Message = llm_response.message

    has_reasoning = (
        message.responses_reasoning_item is not None
        or message.reasoning_content is not None
        or (message.thinking_blocks and len(message.thinking_blocks) > 0)
    )
    has_content = any(
        isinstance(c, TextContent) and c.text.strip() for c in message.content
    )

    if message.tool_calls and len(message.tool_calls) > 0:
        if not all(isinstance(c, TextContent) for c in message.content):
            logger.warning(
                "LLM returned tool calls but message content is not all "
                "TextContent - ignoring non-text content"
            )

        # Generate unique batch ID for this LLM response
        thought_content = [c for c in message.content if isinstance(c, TextContent)]

        action_events: list[ActionEvent] = []
        for i, tool_call in enumerate(message.tool_calls):
            action_event = self._get_action_event(
                tool_call,
                llm_response_id=llm_response.id,
                on_event=on_event,
                security_analyzer=state.security_analyzer,
                thought=thought_content
                if i == 0
                else [],  # Only first gets thought
                # Only first gets reasoning content
                reasoning_content=message.reasoning_content if i == 0 else None,
                # Only first gets thinking blocks
                thinking_blocks=list(message.thinking_blocks) if i == 0 else [],
                responses_reasoning_item=message.responses_reasoning_item
                if i == 0
                else None,
            )
            if action_event is None:
                continue
            action_events.append(action_event)

        # Handle confirmation mode - exit early if actions need confirmation
        if self._requires_user_confirmation(state, action_events):
            return

        if action_events:
            self._execute_actions(conversation, action_events, on_event)

        # Emit VLLM token ids if enabled before returning
        self._maybe_emit_vllm_tokens(llm_response, on_event)
        return

    # No tool calls - emit message event for reasoning or content responses
    if not has_reasoning and not has_content:
        logger.warning("LLM produced empty response - continuing agent loop")

    msg_event = MessageEvent(
        source="agent",
        llm_message=message,
        llm_response_id=llm_response.id,
    )
    on_event(msg_event)

    # Emit VLLM token ids if enabled
    self._maybe_emit_vllm_tokens(llm_response, on_event)


def get_instruction(
    instance: dict,
    metadata: EvalMetadata,
    workspace_path: str,
) -> str:
    """Generate instruction for the agent."""
    workspace_dir_name = instance["repo"].split("/")[-1]
    assert metadata.details is not None

    # Set up Jinja2 environment
    assert metadata.prompt_path is not None
    prompts_dir = os.path.dirname(metadata.prompt_path)
    template_name = os.path.basename(metadata.prompt_path)
    env = Environment(loader=FileSystemLoader(prompts_dir))
    template = env.get_template(template_name)

    # Prepare context for rendering
    context = {
        "instance": instance,
        "workspace_dir_name": workspace_dir_name,
        "actual_workspace_path": workspace_path,
        "metadata": metadata,
        "test_instructions": ""
    }

    # Render the instruction
    instruction = template.render(context)
    return instruction


class SWEBenchEvaluation(Evaluation):
    """
    Process-based SWE-bench evaluation implemented as a child of the
    abstract Evaluation orchestrator.

    Implements:
      - prepare_instances()
      - prepare_workspace(instance)
      - evaluate_instance(instance, workspace)
    """

    def prepare_instances(self) -> List[EvalInstance]:
        logger.info("Setting up SWE-bench evaluation data")

        instances: List[EvalInstance] = [
            make_instance(self.metadata.selected_instances_file)
        ]

        logger.info("Total instances to process: %d", len(instances))
        return instances

    # ---- Hook: prepare a workspace per instance ----------------------------------
    def prepare_workspace(self, instance: EvalInstance) -> RemoteWorkspace:
        """
        Use DockerWorkspace by default.
        """
        if self.metadata.workspace_type == "docker":
            workspace = LocalWorkspace(working_dir=instance.data["repo_path"])
        else:
            raise ValueError(
                f"Unsupported workspace_type: {self.metadata.workspace_type}"
            )

        for cmd in self.metadata.env_setup_commands or []:
            res = workspace.execute_command(cmd)
            if res.exit_code != 0:
                raise RuntimeError(
                    f"Failed to run env setup command '{cmd}': {res.stderr}"
                )
            logger.debug(f"Ran env setup command '{cmd}': {res.stdout}")
        return workspace

    # ---- Hook: evaluate one instance ---------------------------------------------
    def evaluate_instance(
        self, instance: EvalInstance, workspace: RemoteWorkspace
    ) -> EvalOutput:
        """
        Create conversation, run agent, collect history and git patch.
        Do not write files here; just return EvalOutput.
        """
        tools = get_default_tools(
            # Disable browser tools in CLI mode
            enable_browser=False,
        )
        agent = Agent(
            llm=self.metadata.llm,
            tools=tools,
            system_prompt_kwargs={"cli_mode": True},
            # TODO: we can enable condenser and security analyzer later
            # and have them configurable via EvalMetadata
            # condenser=get_default_condenser(
            #     llm=self.metadata.llm.model_copy(update={"service_id": "condenser"})
            # ),
            # security_analyzer=LLMSecurityAnalyzer(),
        )
        # 使用 object.__setattr__ 绕过 Pydantic 的 frozen 限制
        object.__setattr__(agent, "step", types.MethodType(_hijack_step, agent))

        def _log_event(ev):  # keep it simple
            logger.debug("Event: %s", ev)

        conversation = Conversation(
            agent=agent,
            workspace=workspace,
            callbacks=[_log_event],
            max_iteration_per_run=self.metadata.max_iterations,
        )

        repo_path = instance.data["repo_path"]
        proj_path = instance.data["project_path"]
        if proj_path != repo_path:
            cp_repo = workspace.execute_command(
                f"mkdir -p {repo_path} ; rm -rf {repo_path}/* ; cp -r {proj_path}/. {repo_path}"
            )
            assert cp_repo.exit_code == 0, f"cp_repo failed: {cp_repo.stderr}"
        if not instance.data["base_commit"]:
            hash_res = workspace.execute_command(
                f"cd {proj_path} && git rev-parse HEAD"
            )
            assert hash_res.exit_code == 0
            instance.data["base_commit"] = hash_res.stdout.strip()

        # git reset
        git_reset = workspace.execute_command(f"cd {repo_path} ; git reset --hard")
        assert git_reset.exit_code == 0, f"git reset failed: {git_reset.stderr}"

        instruction = get_instruction(
            instance=instance.data,
            metadata=self.metadata,
            workspace_path=workspace.working_dir,
        )
        conversation.send_message(instruction)
        conversation.run()

        # git add
        workspace.execute_command(f"cd {repo_path} ; git add -A")

        # git commit
        workspace.execute_command(
            f"cd {repo_path} && "
            "git config --global user.email 'evaluation@openhands.dev' && "
            "git config --global user.name 'OpenHands Evaluation' && "
            "git commit -m 'patch'"
        )

        # Get git patch
        base_commit = instance.data["base_commit"]
        git_patch_result = workspace.execute_command(
            (f"cd {repo_path} ; git --no-pager diff --no-color {base_commit} HEAD")
        )
        assert git_patch_result.exit_code == 0, (
            f"git diff failed: {git_patch_result.stderr}"
        )
        git_patch = git_patch_result.stdout

        # EvalOutput is your model; keep fields consistent with prior JSONL
        out = EvalOutput(
            instance_id=instance.id,
            test_result={
                "git_patch": git_patch,
            },
            instruction=instruction,
            error=None,
            history=list(conversation.state.events),
            metrics=conversation.conversation_stats.get_combined_metrics(),
        )
        return out


def main() -> None:
    prompt_dir = (Path(__file__).parent / "prompts").resolve()
    choices = [str(p.relative_to(Path.cwd())) for p in prompt_dir.glob("*.j2")]
    default_prompt_path = prompt_dir / "default.j2"
    assert default_prompt_path.exists(), (
        f"Default prompt {default_prompt_path} not found"
    )

    parser = get_parser()
    parser.add_argument(
        "--prompt-path",
        type=str,
        default=str(default_prompt_path),
        choices=choices,
        help="Path to prompt template file",
    )
    args = parser.parse_args()

    # Validate max_attempts
    if args.max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {args.max_attempts}")

    llm_config_path = args.llm_config_path
    if not os.path.isfile(llm_config_path):
        raise ValueError(f"LLM config file {llm_config_path} does not exist")
    with open(llm_config_path, "r") as f:
        llm_config = f.read()
    llm = LLM.model_validate_json(llm_config)
    logger.info("Using LLM config: %s", llm.model_dump_json(indent=2))

    dataset_description = (
        args.dataset.replace("/", "__") + "-" + args.split.replace("/", "__")
    )

    structured_output_dir = construct_eval_output_dir(
        base_dir=args.output_dir,
        dataset_name=dataset_description,
        model_name=llm.model,
        max_iterations=args.max_iterations,
        eval_note=args.note,
    )

    # Create critic instance from parsed arguments
    critic = create_critic(args)
    logger.info(f"Using critic: {type(critic).__name__}")

    metadata = EvalMetadata(
        llm=llm,
        dataset=args.dataset,
        dataset_split=args.split,
        max_iterations=args.max_iterations,
        eval_output_dir=structured_output_dir,
        details={},
        prompt_path=args.prompt_path,
        eval_limit=args.n_limit,
        env_setup_commands=["export PIP_CACHE_DIR=~/.cache/pip"],
        max_attempts=args.max_attempts,
        critic=critic,
        selected_instances_file=args.select,
        max_retries=args.max_retries,
        workspace_type=args.workspace,
    )

    # Run orchestrator with a simple JSONL writer
    evaluator = SWEBenchEvaluation(
        metadata=metadata,
        num_workers=args.num_workers,
    )

    evaluator.run(on_result=get_default_on_result_writer(evaluator.output_path))

    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
