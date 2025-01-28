from abc import ABC, abstractmethod

from learning_frameworks.collection.buffer import AgentResult, StepResult


class Callback(ABC):

    @abstractmethod
    def before_collection(self):
        pass

    @abstractmethod
    def after_collection(self):
        pass

    @abstractmethod
    def before_step(self, agent_result: AgentResult):
        pass

    @abstractmethod
    def after_step(self, step_result: StepResult):
        pass


class EmptyCallback(Callback):
    def before_collection(self):
        pass

    def after_collection(self):
        pass

    def before_step(self, agent_result: AgentResult):
        pass

    def after_step(self, step_result: StepResult):
        pass
