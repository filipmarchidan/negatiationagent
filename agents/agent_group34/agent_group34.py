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
from geniusweb.issuevalue.Value import Value
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


class AgentGroup34(DefaultParty):
    """
    Template of a Python geniusweb agent.
    """

    def __init__(self):
        super().__init__()
        self.logger: ReportToLogger = self.getReporter()

        self.domain: Domain = None
        self.parameters: Parameters = None
        self.profile: LinearAdditiveUtilitySpace = None
        self.progress: ProgressTime = None
        self.me: PartyId = None
        self.other: str = None
        self.settings: Settings = None
        self.storage_dir: str = None

        self.last_received_bid: Bid = None
        self.opponent_model: OpponentModel = None
        self.logger.log(logging.INFO, "party is initialized")

        # New added fields:
        self.all_bids: list[tuple[Bid, float]] = None   # all possible bids in the issue space (with utility above the reservation value)
        self.reservation_value: float = None            # the worst deal we are willing to accept
        self.last_offered_bid: Bid = None               # our previous offer
        self.received_bids = []                         # stores all the bids we received
        self.W: int = 300                               # the time window in AC_combi(MAX^W) (in number of rounds)
        self.target_utility: float = None               # the utility the agent is going for when preparing a bid offer
        self.max_possible_utility: float = 1            # the max utility the agent can achieve in the domain space

    def notifyChange(self, data: Inform):
        """
        The entry point of all interaction with your agent after is has been initialised.
        How to handle the received data is based on its class type.

        Args:
            info (Inform): Contains either a request for action or information.
        """

        # a Settings message is the first message that will be sent to your
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

            # -------------------------------------------
            # The list of all possible bids:
            bids = AllBidsList(self.domain)

            # Save the reservation value:
            reservation_bid = self.profile.getReservationBid()
            if reservation_bid is not None:
                self.reservation_value = float(self.profile.getUtility(reservation_bid))
            else:
                self.reservation_value = float(0)

            # Consider only the bids above the reservation value,
            # and precompute their utility which is contained in a tuple list with their respective bid
            self.all_bids = list(filter(lambda tup: (tup[1] > self.reservation_value), map(lambda b: (b, float(self.profile.getUtility(b))), bids)))

            # Initialize target utility:
            self.max_possible_utility = max(self.all_bids, key=lambda tup: tup[1])[1]
            self.target_utility = self.max_possible_utility
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
        """
        Method to indicate to the protocol what the capabilities of this agent are.

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
        """
        Returns a description of your agent. 1 or 2 sentences.

        Returns:
            str: Agent description
        """
        return "Group 34's negotiation agent."

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

            # add bid to the list of received bids:
            self.received_bids.append(bid)

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
    ################################## OUR IMPLEMENTATION #####################################
    ###########################################################################################

    def accept_condition(self, bid: Bid) -> bool:
        """
        Decide whether to accept or reject the opponent's bid.

        Args:
            bid: the bid that is to be accepted or not
        Returns:
            bool: whether the bid was accepted or not
        """
        if bid is None:
            return False

        # progress of the negotiation session between 0 and 1 (1 is deadline)
        progress = self.progress.get(time() * 1000)

        bid_utility = float(self.profile.getUtility(bid))
        upcoming_bid_utility = float(self.profile.getUtility(self.find_bid()))

        # Accept the offer if *any* of the following conditions are respected:
        conditions = [
            # Received higher utility than our last proposed bid:
            self.last_offered_bid is not None and bid_utility >= float(self.profile.getUtility(self.last_offered_bid)),

            # Received higher utility than our upcoming bid:
            self.last_offered_bid is not None and bid_utility >= upcoming_bid_utility,

            # AC_combi(MAX ^ W) rule (active only when 75% of the time has passed):
            progress > 0.75 and len(self.received_bids) > self.W and self.AC_combi_MAX_W(bid),

            # Negotiation is close to an end:
            progress >= 0.99
        ]

        # + accept the bid only when it is above the reservation_value (hard requirement)
        return (bid_utility > self.reservation_value) and any(conditions)

    def AC_combi_MAX_W(self, bid) -> bool:
        """
        Applies AC_combi(MAX^W) rule to a given bid.
        Basically accepts any offer that is better than any received in the last W rounds.
        """
        bid_utility = float(self.profile.getUtility(bid))

        # Get the max utility in the last W rounds of negotiation:
        # [-(self.W + 1):-1] selects the last W+1 elements from the list and ignores the last one (not taking the current received bid into account)
        utilities_in_time_window_W = list(map(lambda b: float(self.profile.getUtility(b)), self.received_bids[-(self.W + 1):-1]))
        max_utility_in_time_window_W = max(utilities_in_time_window_W)

        return bid_utility > max_utility_in_time_window_W

    def find_bid(self) -> Bid:
        """
        Finds a bid to offer to the opponent.
        """
        progress = self.progress.get(time() * 1000)

        # Randomly pick the best possible bids until 25% of the time has passed
        # We can't propose bids that the opponent likes because we don't know their preferences yet
        if progress < 0.25:
            return self.find_random_best_bid()

        # Compute the difference in the opponent's utility at each time step (sort of a time series)
        received_utilities = list(map(lambda b: (float(self.opponent_model.get_predicted_utility(b))), self.received_bids))
        bid_diffs = [received_utilities[i] - received_utilities[i - 1] for i in range(1, len(received_utilities))]

        # E denotes the concession rate (negative means that we are conceding)
        # We're trying to mimic the opponent's concession rate, i.e. the average(!) difference in opponent's utility of consecutive offered bids:
        E: float = sum(bid_diffs)/len(bid_diffs)
        # If over time the opponent has a positive concession rate (hardliner, pushes for higher gains) then we are going to concede by the amount 1e-04
        E = min(E, -1e-04)

        min_bid_utility = 0.6  # the minimum bid utility we are willing to consider
        # Update the target utility:
        self.target_utility = min([self.max_possible_utility, max([self.target_utility + E, min_bid_utility])])
        # If we reached the minimum, start again from a target utility of 0.8
        if self.target_utility == min_bid_utility:
            self.target_utility = min(self.max_possible_utility, 0.8)

        # The most preferable bid based on the opponent model:
        best_bid = self.find_optimal_bid(self.target_utility)

        # Couldn't find best bid:
        if best_bid is None:
            return self.find_random_best_bid()

        self.last_offered_bid = best_bid
        return best_bid

    def find_random_best_bid(self) -> Bid:
        """
        <Code provided in the original template>
        """
        best_bid_score = 0.0
        best_bid = None

        # take 500 attempts to find a bid according to a heuristic score
        for _ in range(500):
            bid = self.all_bids[randint(0, len(self.all_bids) - 1)][0]
            bid_score = self.score_bid(bid)
            if bid_score > best_bid_score:
                best_bid_score, best_bid = bid_score, bid

        self.last_offered_bid = best_bid
        return best_bid

    def find_optimal_bid(self, target_utility) -> Bid:
        """
        Employs the trade-off bidding strategy and finds the most similar bid to the last received bid, that has the utility
        above a certain threshold.

        Args:
            target_utility: the minimum utility we're aiming for

        Returns:
            Bid: the optimal bid respecting the factors considered
        """
        best_bid = None
        best_bid_utility = 0.0
        best_bid_similarity = 0

        # Iterate over all possible bids:
        for bid, bid_utility in self.all_bids:

            # Consider bids that have utility greater than the aspirational scoring value:
            if bid_utility < target_utility:
                continue

            # Similarity between the bid and the last received bid:
            bid_similarity = self.similarity(bid, self.last_received_bid)

            if bid_similarity == 0:
                continue

            # Choose the most similar bid:
            if (bid_similarity > best_bid_similarity) or (bid_similarity == best_bid_similarity and bid_utility > best_bid_utility):
                best_bid, best_bid_similarity, best_bid_utility = bid, bid_similarity, bid_utility

        return best_bid

    def similarity(self, bid1, bid2) -> float:
        """
        Computes similarity between two bids. Makes use of the opponent model to get the estimated issue weights.

        Returns:
            float: value denoting the similarity between two bids.
        """
        similarity: float = 0

        # Iterate over all issues:
        for issue_id, issue_estimator in self.opponent_model.issue_estimators.items():
            value1: Value = bid1.getValue(issue_id)     # value of the issue in the 1st bid
            value2: Value = bid2.getValue(issue_id)     # value of the issue in the 2nd bid

            # The estimated weight for this issue in the opponent model:
            issue_weight = float(issue_estimator.weight)

            if value1 == value2:
                values_similarity = 1.0
            else:
                values_similarity = 0.0

            similarity += issue_weight * values_similarity

        return similarity

    def score_bid(self, bid: Bid, alpha: float = 0.95, eps: float = 0.1) -> float:
        """
        <Method provided in the original template>
        Calculate heuristic score for a bid

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

