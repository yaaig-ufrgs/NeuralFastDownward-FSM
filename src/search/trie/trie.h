#ifndef TRIE_TRIE_H
#define TRIE_TRIE_H

/**
 * Source: https://github.com/akshitgrover/trie
 */

#include <vector>
#include <assert.h>

#include "trie_node.h"
#include "trie_iterator.h"

enum UpdateRule { vu_u, v_vu, v_v };
UpdateRule getRule(std::string rule) {
  if (rule=="vu_u")
    return UpdateRule::vu_u;
  else if (rule=="v_vu")
    return UpdateRule::v_vu;
  else if (rule=="v_v")
    return UpdateRule::v_v;
  assert(false);
  return UpdateRule::v_v;
}

namespace trie {
  template <typename T> class trie {
  public:
    using iterator = trie_iterator<T>;
    using reverse_iterator = std::reverse_iterator<iterator>;

    trie();
    void insert(std::vector<int>, T);
    bool exist(std::vector<int>);
    bool empty();
    iterator begin();
    iterator end();
    reverse_iterator rbegin();
    reverse_iterator rend();
    iterator find(std::vector<int>);
    void find_all_compatible(std::vector<int> key, UpdateRule rule, std::vector<T>& values);

  private:
    void find_all_compatible_rec(const std::vector<int>& key, unsigned pos, tnode<T>* n, UpdateRule rule, std::vector<T>& values);

    tnode<T> *root;
    int size;
    bool is_empty;
  };

  template <typename T>
  trie<T>::trie(): size(0) {
    T flag = T();
    root = new tnode<T>(flag, nullptr, -1);
    size = 0;
    is_empty = true;
  }

  template <typename T>
  void trie<T>::insert(std::vector<int> key, T val) {
    tnode<T>* node = this->root;
    for (int& v : key) {
      // Our use case has -1, so its increments to get the values in the range (0..MAX_CHILDREN-1)
      v += 1; assert(v >= 0 && v < MAX_CHILDREN);

      if (node->getChild(v) != nullptr) {
	node = node->getChild(v);
      } else {
	T flag = T();
	tnode<T>* _node = new tnode<T>(flag, node, v);
	node->addChild(_node, v);
	node = node->getChild(v);
      }
    }
    if (!node->isEnd()) {
      this->size += 1;
    }
    this->is_empty = false;
    node->update(val);
    node->markEnd(key);
  }

  template <typename T>
  bool trie<T>::exist(std::vector<int> key) {
    bool res = true;
    tnode<T>* node = this->root;
    for (int& v : key) {
      // Our use case has -1, so its increments to get the values in the range (0..MAX_CHILDREN-1)
      v += 1; assert(v >= 0 && v < MAX_CHILDREN);

      if (node->getChild(v) == nullptr) {
	res = false;
	break;
      } else {
	node = node->getChild(v);
      }
    }
    if (!node->isEnd()) {
      res = false;
    }
    return res;
  }

  template <typename T>
  bool trie<T>::empty() {
    // return this->size == 0;
    return this->is_empty;
  }

  template <typename T>
  typename trie<T>::iterator trie<T>::begin() {
    trie_iterator<T> it = *(new trie_iterator<T>(this->root));
    return ++it;
  }

  template <typename T>
  tnode<T>* rbrecur(tnode<T>* n, int offset = MAX_CHILDREN-1, tnode<T>* r = nullptr);

  template <typename T>
  typename trie<T>::iterator trie<T>::end() {
    T flag;
    tnode<T>* r = nullptr;
    if (!this->empty()) {
      r = rbrecur(this->root);
    }
    tnode<T>* t = new tnode<T>(flag, r, 1516);
    return *(new trie_iterator<T>(t));
  }

  template <typename T>
  tnode<T>* rbrecur(tnode<T>* n, int offset, tnode<T>* r) {
    tnode<T>* it = nullptr;
    for (int i = offset; i > -1; i--) {
      it = n->getChild(i);
      if (it == nullptr) {
	if (i == 0) {
	  return r;
	}
	continue;
      }
      if (it->isEnd()) {
	r = it;
      }
      return rbrecur(it, MAX_CHILDREN-1, r);
    }
    return nullptr;
  }

  template <typename T>
  typename trie<T>::reverse_iterator trie<T>::rbegin() {
    return *(new trie<T>::reverse_iterator(trie<T>::end()));
  }

  template <typename T>
  typename trie<T>::reverse_iterator trie<T>::rend() {
    return *(new trie<T>::reverse_iterator(trie<T>::begin()));
  }

  template <typename T>
  typename trie<T>::iterator trie<T>::find(std::vector<int> key) {
    tnode<T>* n = this->root;
    for (int& v : key) {
      // Our use case has -1, so its increments to get the values in the range (0..MAX_CHILDREN-1)
      v += 1; assert(v >= 0 && v < MAX_CHILDREN);

      n = n->getChild(v);
      if (n == nullptr) {
	return this->end();
      }
    }
    if (!n->isEnd()) {
      return this->end();
    }
    trie_iterator<T> it = *(new trie_iterator<T>(n));
    return it;
  }

  template <typename T>
  void trie<T>::find_all_compatible(std::vector<int> key, UpdateRule rule, std::vector<T>& values) {
    // Our use case has -1, so its increments to get the values in the range (0..MAX_CHILDREN-1)
    for (int& v : key) {
      v++;
      assert(v >= 0 && v < MAX_CHILDREN);
    }
    find_all_compatible_rec(key, 0, this->root, rule, values);
  }

  template <typename T>
  void trie<T>::find_all_compatible_rec(const std::vector<int>& key, unsigned pos, tnode<T>* n, UpdateRule rule, std::vector<T>& values) {
    if (n==nullptr)
      return;
  
    if (pos==key.size()) {
      values.push_back(n->get());
      return;
    }
    
    find_all_compatible_rec(key, pos + 1, n->getChild(key[pos]), rule, values);

    // let 0 = undefined, v = any other value
    if (rule == UpdateRule::vu_u) {
      // v -> v || u
      // u -> u
      if (key[pos] != 0) // if (key[pos] == v)
	find_all_compatible_rec(key, pos + 1, n->getChild(0), rule, values);
    } else if (rule == UpdateRule::v_vu) {
      // v -> v
      // u -> v || u
      if (key[pos] == 0) // if (key[pos] == u)
	for (int i = 1; i < MAX_CHILDREN-1; i++)
	  find_all_compatible_rec(key, pos + 1, n->getChild(i), rule, values);
    }
  }
} // namespace trie

#endif
