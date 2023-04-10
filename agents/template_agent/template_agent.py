import logging
from random import randint
from time import time
from typing import cast

from geniusweb.actions.Accept import Accept
from geniusweb.actions.Action import Action
from geniusweb.actions.Offer import Offer
from geniusweb.actions.PartyId import PartyId
from geniusweb.bidspace.AllBidsList import AllBidsList
from geniusweb.inform.ActionDone import ActionDone
from geniusweb.inform.Finished import Finished
from geniusweb.inform.Inform import Inform
from geniusweb.inform.Settings import Settings
from geniusweb.inform.YourTurn import YourTurn
from geniusweb.issuevalue.Bid import Bid
from geniusweb.issuevalue.Domain import Domain
from geniusweb.party.Capabilities import Capabilities
from geniusweb.party.DefaultParty import DefaultParty
from geniusweb.profile.utilityspace.LinearAdditiveUtilitySpace import (
    LinearAdditiveUtilitySpace,
)
from geniusweb.profileconnection.ProfileConnectionFactory import (
    ProfileConnectionFactory,
)
from geniusweb.progress.ProgressTime import ProgressTime
from geniusweb.references.Parameters import Parameters
from tudelft_utilities_logging.ReportToLogger import ReportToLogger

from .utils.opponent_model import OpponentModel


class TemplateAgent(DefaultParty):
    """
    Template of a Python geniusweb agent.
    """

    def __init__(self):
        super().__init__()
        self.preferred_utility: float = None
        self.logger: ReportToLogger = self.getReporter()

        self.domain: Domain = None
        self.parameters: Parameters = None
        self.profile: LinearAdditiveUtilitySpace = None
        self.progress: ProgressTime = None
        self.me: PartyId = None
        self.other: str = None
        self.settings: Settings = None
        self.storage_dir: str = None

        self.last_received_bid: Bid = None  # Used in AcNext
        self.last_offered_bid: Bid = None
        self.all_bids_filtered = None
        self.opponent_model: OpponentModel = None
        self.first_window_bids: [Bid] = []  # Keeps track of opponent's bids from first time window
        self.size_of_first_time_window = 0.75  # Defines the size of first time window of opponent's bids
        self.logger.log(logging.INFO, "party is initialized")
        self.reservation_value: float = None

    def notifyChange(self, data: Inform):
        """MUST BE IMPLEMENTED
        This is the entry point of all interaction with your agent after is has been initialised.
        How to handle the received data is based on its class type.

        Args:
            info (Inform): Contains either a request for action or information.
        """

        # a Settings message is the first message that will be send to your
        # agent containing all the information about the negotiation session.
        if isinstance(data, Settings):
            self.settings = cast(Settings, data)
            self.me = self.settings.getID()

            # progress towards the deadline has to be tracked manually through the use of the Progress object
            self.progress = self.settings.getProgress()

            self.parameters = self.settings.getParameters()
            self.storage_dir = self.parameters.get("storage_dir")

            # the profile contains the preferences of the agent over the domain
            profile_connection = ProfileConnectionFactory.create(
                data.getProfile().getURI(), self.getReporter()
            )
            self.profile = profile_connection.getProfile()
            self.domain = self.profile.getDomain()

            reservation_bid = self.profile.getReservationBid()
            if reservation_bid is not None:
                self.reservation_value = float(self.profile.getUtility(reservation_bid))
            else:
                self.reservation_value = float(0.4)

            domain = self.profile.getDomain()
            all_bids = AllBidsList(domain)
            self.all_bids_filtered = list(
                filter(lambda b: (float(self.profile.getUtility(b)) >= self.reservation_value), all_bids))
            profile_connection.close()

        # ActionDone informs you of an action (an offer or an accept)
        # that is performed by one of the agents (including yourself).
        elif isinstance(data, ActionDone):
            action = cast(ActionDone, data).getAction()
            actor = action.getActor()

            # ignore action if it is our action
            if actor != self.me:
                # obtain the name of the opponent, cutting of the position ID.
                self.other = str(actor).rsplit("_", 1)[0]

                # process action done by opponent
                self.opponent_action(action)
        # YourTurn notifies you that it is your turn to act
        elif isinstance(data, YourTurn):
            # execute a turn
            self.my_turn()

        # Finished will be send if the negotiation has ended (through agreement or deadline)
        elif isinstance(data, Finished):
            self.save_data()
            # terminate the agent MUST BE CALLED
            self.logger.log(logging.INFO, "party is terminating:")
            super().terminate()
        else:
            self.logger.log(logging.WARNING, "Ignoring unknown info " + str(data))

    def getCapabilities(self) -> Capabilities:
        """MUST BE IMPLEMENTED
        Method to indicate to the protocol what the capabilities of this agent are.
        Leave it as is for the ANL 2022 competition

        Returns:
            Capabilities: Capabilities representation class
        """
        return Capabilities(
            set(["SAOP"]),
            set(["geniusweb.profile.utilityspace.LinearAdditive"]),
        )

    def send_action(self, action: Action):
        """Sends an action to the opponent(s)

        Args:
            action (Action): action of this agent
        """
        self.getConnection().send(action)

    # give a description of your agent
    def getDescription(self) -> str:
        """MUST BE IMPLEMENTED
        Returns a description of your agent. 1 or 2 sentences.

        Returns:
            str: Agent description
        """
        return "Template agent for the ANL 2022 competition"

    def opponent_action(self, action):
        """Process an action that was received from the opponent.

        Args:
            action (Action): action of opponent
        """
        # if it is an offer, set the last received bid
        if isinstance(action, Offer):
            # create opponent model if it was not yet initialised
            if self.opponent_model is None:
                self.opponent_model = OpponentModel(self.domain)

            bid = cast(Offer, action).getBid()

            # update opponent model with bid
            self.opponent_model.update(bid)
            # set bid as last received
            self.last_received_bid = bid
            progress = self.progress.get(time() * 1000)
            if progress <= self.size_of_first_time_window:
                self.first_window_bids.append(self.last_received_bid)

    def my_turn(self):
        """This method is called when it is our turn. It should decide upon an action
        to perform and send this action to the opponent.
        """
        # check if the last received offer is good enough

        if self.accept_condition(self.last_received_bid):
            # if so, accept the offer

            action = Accept(self.me, self.last_received_bid)
        else:
            # if not, find a bid to propose as counter offer
            bid = self.find_bid()
            self.last_offered_bid = bid
            action = Offer(self.me, bid)

        # send the action
        self.send_action(action)

    def save_data(self):
        """This method is called after the negotiation is finished. It can be used to store data
        for learning capabilities. Note that no extensive calculations can be done within this method.
        Taking too much time might result in your agent being killed, so use it for storage only.
        """
        data = "Data for learning (see README.md)"
        with open(f"{self.storage_dir}/data.md", "w") as f:
            f.write(data)

    ###########################################################################################
    ################################## Example methods below ##################################
    ###########################################################################################

    # Accept if the utility of the received bid is higher than next utility offered bid
    def accept_next(self, opp_bid: Bid) -> bool:
        next_bid = self.find_bid()
        # TODO change or tune in params
        alpha = 1
        beta = 0
        opp_bid_utility = self.profile.getUtility(opp_bid)
        next_bid_utility = self.profile.getUtility(next_bid)
        if alpha * opp_bid_utility + beta >= next_bid_utility:
            return True
        return False

    # Accept if the utility of the received bid is higher than the last offered bid
    def accept_previous(self, opp_bid: Bid) -> bool:
        alpha = 1
        beta = 0
        opp_bid_utility = self.profile.getUtility(opp_bid)
        if self.last_offered_bid is not None:
            prev_bid_utility = self.profile.getUtility(self.last_offered_bid)
            if alpha * opp_bid_utility + beta >= prev_bid_utility:
                return True
        return False

    # Accept if round progress is at 95%
    def accept_time(self) -> bool:
        time_const = 0.99
        progress = self.progress.get(time() * 1000)
        if progress >= time_const:
            return True
        return False

    # Accept if the current opponent's bid utility is higher than the average utility seen in the first time window
    def accept_max(self, opp_bid: Bid) -> bool:
        progress = self.progress.get(time() * 1000)

        if progress < self.size_of_first_time_window:
            return False
        average_opp_utility = self.compute_utility_average_of_first_window()

        if self.profile.getUtility(opp_bid) >= average_opp_utility:
            return True
        return False

    # It computes the average of utilities of opponent's bids that were received in the first time window
    # First time window is defined as follows: [0, self.size_of_first_time_window]
    def compute_utility_average_of_first_window(self) -> float:
        current = 0
        max = 0
        for opp_bid in self.first_window_bids:
            if current < self.profile.getUtility(opp_bid):
                max = self.profile.getUtility(opp_bid)
            else:
                max = current
        return max

    def accept_condition(self, bid: Bid) -> bool:
        if bid is None:
            return False

            # TODO change equation if needed
        #print(self.profile.getUtility(bid), "bid utility")
        #print(self.reservation_value, "reservation value")
        return (self.profile.getUtility(bid) > self.reservation_value) and (
            (self.accept_time() or (self.accept_max(bid))) or self.accept_previous(bid) and self.accept_next(bid))

    def find_bid(self) -> Bid:
        progress = self.progress.get(time() * 1000)
        # compose a list of all possible bids
        best_bid_score = 0.0
        best_bid = None

        if progress > 0.2:

            if self.preferred_utility is None:
                self.preferred_utility = float(self.profile.getUtility(self.last_offered_bid))

            if (
                    best_bid is None or best_bid == self.last_offered_bid) and self.preferred_utility > self.reservation_value:

                self.preferred_utility = max(self.preferred_utility - 0.01, self.reservation_value)

                if self.preferred_utility > self.reservation_value:
                    return self.find_bid()

            self.last_offered_bid = best_bid
            # return best_bid
        # take 500 attempts to find a bid according to a heuristic score
        for _ in range(500):
            bid = self.all_bids_filtered[(randint(0, len(self.all_bids_filtered) - 1))]
            bid_score = self.score_bid(bid)
            if bid_score > best_bid_score and bid_score > self.reservation_value:
                best_bid_score, best_bid = bid_score, bid

        self.last_offered_bid = best_bid
        return best_bid

    def score_bid(self, bid: Bid, alpha: float = 0.95, eps: float = 0.1) -> float:
        """Calculate heuristic score for a bid

        Args:
            bid (Bid): Bid to score
            alpha (float, optional): Trade-off factor between self interested and
                altruistic behaviour. Defaults to 0.95.
            eps (float, optional): Time pressure factor, balances between conceding
                and Boulware behaviour over time. Defaults to 0.1.

        Returns:
            float: score
        """
        progress = self.progress.get(time() * 1000)

        our_utility = float(self.profile.getUtility(bid))

        time_pressure = 1.0 - progress ** (1 / eps)
        score = alpha * time_pressure * our_utility

        if self.opponent_model is not None:
            opponent_utility = self.opponent_model.get_predicted_utility(bid)
            opponent_score = (1.0 - alpha * time_pressure) * opponent_utility
            score += opponent_score

        return score
