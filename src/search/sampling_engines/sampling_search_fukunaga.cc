#include "sampling_search_fukunaga.h"

#include "sampling_search_base.h"
#include "sampling_engine.h"

#include "../option_parser.h"
#include "../plugin.h"

#include "../task_utils/task_properties.h"

#include <sstream>
#include <string>

using namespace std;

namespace sampling_engine {

string SamplingSearchFukunaga::construct_header() const {
    ostringstream oss;

    if (store_plan_cost){
        oss << "#<PlanCost>=single integer value" << endl;
    }
    if (store_state) {
        oss << "#<State>=";
        for (unsigned i = 0; i < relevant_facts.size(); i++) {
            if ((state_representation == "undefined") &&
                (i == 0 || relevant_facts[i].var != relevant_facts[i-1].var))
                oss << "Atom undefined()" << state_separator;
            oss << task->get_fact_name(relevant_facts[i]) << state_separator;
        }
        oss.seekp(-1,oss.cur);
    }

    return oss.str();
}

string SamplingSearchFukunaga::sample_file_header() const {
    return header;
}

vector<string> SamplingSearchFukunaga::extract_samples() {
    vector<string> samples;
    int rand_value = 0;
    int max_h = 0;

    for (std::shared_ptr<PartialAssignment>& partialAssignment: sampling_technique::modified_tasks) {
        ostringstream oss;

        int h = partialAssignment->estimated_heuristic;
        if (h > max_h)
            max_h = h;

        if (store_plan_cost && state_representation != "assign_undefined")
            oss << h << field_separator;

        if (store_state) {
            vector<int> values;
            if (state_representation == "assign_undefined") {
                vector<State> states;
                for (int attempts = 0, assignments = 0;
                     attempts < assignments_by_undefined_state*5 && assignments < assignments_by_undefined_state;
                     attempts++
                ) {
                    utils::RandomNumberGenerator rand(rand_value++);
                    pair<bool,State> full_state = partialAssignment->get_full_state(true, rand);
                    if (!full_state.first)
                        continue;
                    State s = full_state.second;
                    if (count(states.begin(), states.end(), s) == 0) {
                        states.push_back(s);
                        assignments++;
                        attempts = 0;
                    }
                }
                for (State &s : states) {
                    oss.str("");
                    if (store_plan_cost)
                        oss << partialAssignment->estimated_heuristic << field_separator;
                    s.unpack();
                    values = s.get_values();
                    for (unsigned i = 0; i < relevant_facts.size(); i++)
                        oss << (values[relevant_facts[i].var] == relevant_facts[i].value ? 1 : 0);
                    samples.push_back(oss.str());
                }
            } else {
                if (state_representation == "complete") {
                    State s = partialAssignment->get_full_state(true, *rng).second;
                    s.unpack();
                    values = s.get_values();
                } else if (state_representation == "partial" || state_representation == "undefined") {
                    values = partialAssignment->get_values();
                }
                for (unsigned i = 0; i < relevant_facts.size(); i++) {
                    if ((state_representation == "undefined") &&
                        (i == 0 || relevant_facts[i].var != relevant_facts[i-1].var))
                        oss << (values[relevant_facts[i].var] == PartialAssignment::UNASSIGNED);
                    oss << (values[relevant_facts[i].var] == relevant_facts[i].value ? 1 : 0);
                }
                samples.push_back(oss.str());
            }
        }
    }

    // If match_heuristics then the h values will reduce. Find the new max h.
    if (match_heuristics) {
        unordered_map<string,int> pairs;
        string state;
        int h, cut;
        for (string& s : samples) {
            cut = s.find_first_of(";");
            h = stoi(s.substr(0, cut));
            string state = s.substr(cut+1);
            if (pairs.count(state) == 0 || h < pairs[state])
                pairs[state] = h;
        }
        max_h = 0;
        for (auto& it : pairs)
            if (it.second > max_h)
                max_h = it.second;
    }

    if (contrasting_samples > 0) {
        assert(state_representation != "assign_undefined");

        const size_t n_atoms = sampling_technique::modified_tasks[0]->get_values().size();
        PartialAssignment pa(
            *sampling_technique::modified_tasks[0],
            vector<int>(n_atoms, PartialAssignment::UNASSIGNED)
        );

        assert(contrasting_samples >= 0 && contrasting_samples <= 100);
        int random_samples;
        if (contrasting_samples == 100) {
            // TODO: if 100%, skip the samples generation step
            random_samples = samples.size();
            samples.clear();
        } else {
            random_samples = (sampling_technique::modified_tasks.size() * contrasting_samples) / (100.0 - contrasting_samples);
        }
        while (random_samples > 0) {
            pair<bool,State> fs = pa.get_full_state(true, *rng);
            if (!fs.first)
                continue;
            State s = fs.second;
            s.unpack();
            vector<int> values = s.get_values();

            ostringstream oss;
            oss << (max_h + 1) << field_separator;
            for (unsigned i = 0; i < relevant_facts.size(); i++) {
                if ((state_representation == "undefined") &&
                    (i == 0 || relevant_facts[i].var != relevant_facts[i-1].var))
                    oss << (values[relevant_facts[i].var] == PartialAssignment::UNASSIGNED);
                oss << (values[relevant_facts[i].var] == relevant_facts[i].value ? 1 : 0);
            }
            samples.push_back(oss.str());
            random_samples--;
        }
    }

    if (match_heuristics) {
        unordered_map<string,int> pairs;
        string state;
        int h, cut;
        for (string& s : samples) {
            cut = s.find_first_of(";");
            h = stoi(s.substr(0, cut));
            string state = s.substr(cut+1);
            if (pairs.count(state) == 0 || h < pairs[state])
                pairs[state] = h;
        }
        for (string& s : samples) {
            cut = s.find_first_of(";");
            string state = s.substr(cut+1);
            s = to_string(pairs[state]) + ";" + state;
        }
    }
    return samples;
}

SamplingSearchFukunaga::SamplingSearchFukunaga(const options::Options &opts)
    : SamplingSearchBase(opts),
      store_plan_cost(opts.get<bool>("store_plan_cost")),
      store_state(opts.get<bool>("store_state")),
      state_representation(opts.get<string>("state_representation")),
      match_heuristics(opts.get<bool>("match_heuristics")),
      assignments_by_undefined_state(opts.get<int>("assignments_by_undefined_state")),
      contrasting_samples(opts.get<int>("contrasting_samples")),
      relevant_facts(task_properties::get_strips_fact_pairs(task.get())),
      header(construct_header()),
      rng(utils::parse_rng_from_options(opts)) {
}

static shared_ptr<SearchEngine> _parse_sampling_search_fukunaga(OptionParser &parser) {
    parser.document_synopsis("Sampling Search Manager", "");

    sampling_engine::SamplingSearchBase::add_sampling_search_base_options(parser);
    sampling_engine::SamplingEngine::add_sampling_options(parser);
    sampling_engine::SamplingStateEngine::add_sampling_state_options(
            parser, "fields", "pddl", ";", ";");

    parser.add_option<bool>(
            "store_plan_cost",
            "Store for every state its cost along the plan to the goal",
            "true");
    parser.add_option<bool>(
            "store_state",
            "Store every state along the plan",
            "true");
    parser.add_option<string>(
            "state_representation",
            "State facts representation format (complete, partial, or undefined, assign_undefined).",
            "complete");
    parser.add_option<bool>(
            "match_heuristics",
            "Identical states receive the best heuristic value assigned between them.",
            "false");
    parser.add_option<int>(
            "assignments_by_undefined_state",
            "Number of states generated from each undefined state (only with assign_undefined).",
            "10"
    );
    parser.add_option<int>(
            "contrasting_samples",
            "Generate new random samples with h = L+1. (Percentage of those obtained with the search).",
            "0"
    );

    SearchEngine::add_options_to_parser(parser);
    Options opts = parser.parse();
    shared_ptr<sampling_engine::SamplingSearchFukunaga> engine;
    if (!parser.dry_run()) {
        engine = make_shared<sampling_engine::SamplingSearchFukunaga>(opts);
    }

    return engine;
}

static Plugin<SearchEngine> _plugin_search("sampling_search_fukunaga", _parse_sampling_search_fukunaga);

}
