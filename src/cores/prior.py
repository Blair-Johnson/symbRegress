"""Class for Prior object."""
import numpy as np
import warnings
import inspect
from collections import defaultdict

from cores.library import TokenNotFoundError
from cores.subroutines import ancestors
from cores.subroutines import jit_check_constraint_violation, \
        jit_check_constraint_violation_descendant_with_target_tokens, \
        jit_check_constraint_violation_descendant_no_target_tokens, \
        jit_check_constraint_violation_uchild, get_position, get_mask
from cores.program import Program
# from language_model import LanguageModelPrior as LM
from helpers.utils import import_custom_source


def make_prior(library, config_prior):
    """Factory function for JointPrior object."""

    prior_dict = {
        "length" : LengthConstraint,
    }

    count_constraints = config_prior.pop("count_constraints", False)

    priors = []
    warn_messages = []
    for prior_type, prior_args in config_prior.items():
        if prior_type in prior_dict:
            prior_class = prior_dict[prior_type]
        else:
            # Tries to import custom priors
            prior_class = import_custom_source(prior_type)

        if isinstance(prior_args, dict):
            prior_args = [prior_args]
        for single_prior_args in prior_args:
            # Attempt to build the Prior. Any Prior can fail if it references a
            # Token not in the Library.
            prior_is_enabled = single_prior_args.pop('on', False)
            if prior_is_enabled:
                try:
                    prior = prior_class(library, **single_prior_args)
                    warn_message = prior.validate()
                except TokenNotFoundError:
                    prior = None
                    warn_message = "Uses Tokens not in the Library."
            else:
                prior = None
                warn_message = "Prior disabled."

            # Add warning context
            if warn_message is not None:
                warn_message = "Skipping invalid '{}' with arguments {}. " \
                    "Reason: {}" \
                    .format(prior_class.__name__, single_prior_args, warn_message)
                warn_messages.append(warn_message)

            # Add the Prior if there are no warnings
            if warn_message is None:
                priors.append(prior)

    joint_prior = JointPrior(library, priors, count_constraints)

    print("-- BUILDING PRIOR START -------------")
    print("\n".join(["WARNING: " + message for message in warn_messages]))
    print(joint_prior.describe())
    print("-- BUILDING PRIOR END ---------------\n")

    return joint_prior


class JointPrior():
    """A collection of joint Priors."""

    def __init__(self, library, priors, count_constraints=False):
        """
        Parameters
        ----------
        library : Library
            The Library associated with the Priors.

        priors : list of Prior
            The individual Priors to be joined.

        count_constraints : bool
            Whether to count the number of constrained tokens.
        """

        self.library = library
        self.L = self.library.L
        self.priors = priors
        assert all([prior.library is library for prior in priors]), \
            "All Libraries must be identical."

        # Name the priors, e.g. RepeatConstraint-0
        counts = defaultdict(lambda : -1)
        self.names = []
        for prior in self.priors:
            name = prior.__class__.__name__
            counts[name] += 1
            self.names.append("{}-{}".format(name, counts[name]))

        # Initialize variables for constraint count report
        self.do_count = count_constraints
        self.constraint_indices = [i for i, prior in enumerate(self.priors) if isinstance(prior, Constraint)]
        self.constraint_counts = [0] * len(self.constraint_indices)
        self.total_constraints = 0
        self.total_tokens = 0

        self.requires_parents_siblings = True 

        self.describe()

    def initial_prior(self):
        combined_prior = np.zeros((self.L,), dtype=np.float32)
        for prior in self.priors:
            combined_prior += prior.initial_prior()
        return combined_prior

    def __call__(self, actions, parent, sibling, dangling):
        # Sum the individual priors
        zero_prior = np.zeros((actions.shape[0], self.L), dtype=np.float32)
        ind_priors = [zero_prior.copy() for _ in range(len(self.priors))]
        for i in range(len(self.priors)):
            ind_priors[i] += self.priors[i](actions, parent, sibling, dangling)
        combined_prior = sum(ind_priors) + zero_prior 

        # Count number of constrained tokens per prior
        if self.do_count:
            mask = dangling > 0 
            self.total_tokens += mask.sum() * actions.shape[1]
            for i in self.constraint_indices:
                self.constraint_counts[i] += np.count_nonzero(ind_priors[i][mask] == -np.inf)
            self.total_constraints += np.count_nonzero(combined_prior[mask] == -np.inf)

        return combined_prior

    def report_constraint_counts(self):
        if not self.do_count:
            return
        print("Constraint counts per prior:")
        for i, count in zip(self.constraint_indices, self.constraint_counts):
            print("{}: {} ({:%})".format(self.names[i], count, count / self.total_tokens))
        print("All constraints: {} ({:%})".format(self.total_constraints, self.total_constraints / self.total_tokens))

    def describe(self):
        message = "\n".join(prior.describe() for prior in self.priors)
        return message

    def is_violated(self, actions, parent, sibling):
        for prior in self.priors:
            if isinstance(prior, Constraint):
                if prior.is_violated(actions, parent, sibling):
                    return True
        return False

    def at_once(self, actions, parent, sibling):
        """
        Given a full sequence of actions, parents, and siblings, each of shape
        (batch, time), *retrospectively* compute what was the joint prior at all
        time steps. The combined prior has shape (batch, time, L).
        """

        B, T = actions.shape
        zero_prior = np.zeros((B, T, self.L), dtype=np.float32) # (batch, time, L)
        ind_priors = [zero_prior.copy() for _ in range(len(self.priors))] # i x (batch, time, L)

        # Set initial prior
        # Note: intial_prior() is already a combined prior, so we just set the
        # first individual prior, ind_priors[0].
        initial_prior = self.initial_prior() # Shape (L,)
        ind_priors[0][:, 0, :] = initial_prior # Broadcast to (batch, L)

        dangling = np.ones(B)
        for t in range(1, T): # For each time step
            # Update dangling based on previously sampled token
            dangling += self.library.arities[actions[:, (t - 1)]] - 1
            for i in range(len(self.priors)): # For each Prior
                # Compute the ith Prior at time step t
                prior = self.priors[i](actions[:, :t],
                                       parent[:, t],
                                       sibling[:, t],
                                       dangling) # Shape (batch, L)
                ind_priors[i][:, t, :] += prior

        # Combine all Priors
        combined_prior = sum(ind_priors) + zero_prior

        return combined_prior


class Prior():
    """Abstract class whose call method return logits."""

    def __init__(self, library):
        self.library = library
        self.L = library.L
        self.mask_val = -np.inf

    def validate(self):
        """
        Determine whether the Prior has a valid configuration. This is useful
        when other algorithmic parameters may render the Prior degenerate. For
        example, having a TrigConstraint with no trig Tokens.

        Returns
        -------
        message : str or None
            Error message if Prior is invalid, or None if it is valid.
        """

        return None

    def init_zeros(self, actions):
        """Helper function to generate a starting prior of zeros."""

        batch_size = actions.shape[0]
        prior = np.zeros((batch_size, self.L), dtype=np.float32)
        return prior

    def initial_prior(self):
        """
        Compute the initial prior, before any actions are selected.

        Returns
        -------
        initial_prior : array
            Initial logit adjustment before actions are selected. Shape is
            (self.L,) as it will be broadcast to batch size later.
        """

        return np.zeros((self.L,), dtype=np.float32)

    def __call__(self, actions, parent, sibling, dangling):
        """
        Compute the prior (logit adjustment) given the current actions.

        Returns
        -------
        prior : array
            Logit adjustment for selecting next action. Shape is (batch_size,
            self.L).
        """

        raise NotImplementedError
        
    def describe(self):
        """Describe the Prior."""

        return "{}: No description available.".format(self.__class__.__name__)


class Constraint(Prior):
    def __init__(self, library):
        Prior.__init__(self, library)

    def make_constraint(self, mask, tokens):
        """
        Generate the prior for a batch of constraints and the corresponding
        Tokens to constrain.

        For example, with L=5 and tokens=[1,2], a constrained row of the prior
        will be: [0.0, -np.inf, -np.inf, 0.0, 0.0].

        Parameters
        __________

        mask : np.ndarray, shape=(?,), dtype=np.bool_
            Boolean mask of samples to constrain.

        tokens : np.ndarray, dtype=np.int32
            Tokens to constrain.

        Returns
        _______

        prior : np.ndarray, shape=(?, L), dtype=np.float32
            Logit adjustment. Since these are hard constraints, each element is
            either 0.0 or -np.inf.
        """
        prior = np.zeros((mask.shape[0], self.L), dtype=np.float32)
        
        for t in tokens:
            prior[mask, t] = self.mask_val
        return prior
    
    def is_violated(self, actions, parent, sibling):
        """
        Given a set of actions, tells us if a prior constraint has been violated 
        post hoc. 
        
        This is a generic version that will run using the __call__ function so that one
        does not have to write a function twice for both DSO and Deap. 
        
        >>>HOWEVER<<<
        
        Using this function is less optimal than writing a variant for Deap. So...
        
        If you create a constraint and find you use if often with Deap, you should gp ahead anf
        write the optimal version. 

        Returns
        -------
        violated : Bool
        """
        caller          = inspect.getframeinfo(inspect.stack()[1][0])
        
        warnings.warn("{} ({}) {} : Using a slower version of constraint for Deap. You should write your own.".format(caller.filename, caller.lineno, type(self).__name__))
        
        assert len(actions.shape) == 2, "Only takes in one action at a time since this is how Deap will use it."
        assert actions.shape[0] == 1, "Only takes in one action at a time since this is how Deap will use it."
        
        self.mask_val   = 1.0
        dangling        = np.ones((1), dtype=np.int32)
        
        # For each step in time, get the prior                                
        for t in range(actions.shape[1]):
            dangling    += self.library.arities[actions[:,t]] - 1   
            priors      = self.__call__(actions[:,:t], parent[:,t], sibling[:,t], dangling)
            
            # Does our action conflict with this prior?
            if priors[0,actions[0,t]]:
                return True
             
        return False
    
    def test_is_violated(self, actions, parent, sibling):
        r"""
            This allows one to call the generic version of "is_violated" for testing purposes
            from the derived classes even if they have an optimized version. 
        """
        return Constraint.is_violated(self, actions, parent, sibling)
    
class LengthConstraint(Constraint):
    """Class that constrains the Program from falling within a minimum and/or
    maximum length"""

    def __init__(self, library, min_=None, max_=None):
        """
        Parameters
        ----------
        min_ : int or None
            Minimum length of the Program.

        max_ : int or None
            Maximum length of the Program.
        """

        Prior.__init__(self, library)
        self.min = min_
        self.max = max_
        self.n_objects = Program.n_objects
        if self.n_objects > 1:
            assert self.max is not None, "Is max length constraint turned on? Max length constraint is required when n_objects > 1."

        assert min_ is not None or max_ is not None, \
            "At least one of (min_, max_) must not be None."

    def initial_prior(self):
        prior = Prior.initial_prior(self)
        for t in self.library.terminal_tokens:
            prior[t] = -np.inf
        return prior

    def __call__(self, actions, parent, sibling, dangling):

        # Initialize the prior
        prior = self.init_zeros(actions)
        i = actions.shape[1] - 1 # Current time

        if self.n_objects > 1:
            i, _ = get_position(actions, self.library.arities, n_objects=self.n_objects)

            if self.max is not None:
                remaining = self.max - (i + 1)
                mask = dangling >= remaining - 1 # constrain binary
                prior += self.make_constraint(mask, self.library.binary_tokens)
                mask = dangling == remaining # constrain unary
                prior += self.make_constraint(mask, self.library.unary_tokens)

            # Constrain terminals when dangling == 1 until selecting the
            # (min_length)th token
            if self.min is not None:
                mask = np.logical_and((i + 2) < self.min,
                                        dangling == 1)
                prior += self.make_constraint(mask, self.library.terminal_tokens)
        else:
            # Never need to constrain max length for first half of expression
            if self.max is not None and (i + 2) >= self.max // 2:
                remaining = self.max - (i + 1)
                # assert sum(dangling > remaining) == 0, (dangling, remaining)
                mask = dangling >= remaining - 1 # Constrain binary
                prior += self.make_constraint(mask, self.library.binary_tokens)
                mask = dangling == remaining # Constrain unary
                prior += self.make_constraint(mask, self.library.unary_tokens)

            # Constrain terminals when dangling == 1 until selecting the
            # (min_length)th token
            if self.min is not None and (i + 2) < self.min:
                mask = dangling == 1 # Constrain terminals
                prior += self.make_constraint(mask, self.library.terminal_tokens)

        return prior

    def is_violated(self, actions, parent, sibling):
        l = len(actions[0])
        if self.min is not None and l < self.min:
            return True
        if self.max is not None and l > self.max:
            return True

        return False

    def describe(self):
        message = []
        indent = " " * len(self.__class__.__name__) + "  "
        if self.min is not None:
            message.append("{}: Sequences have minimum length {}.".format(self.__class__.__name__, self.min))
        if self.max is not None:
            message.append(indent + "Sequences have maximum length {}.".format(self.max))
        message = "\n".join(message)
        return message
