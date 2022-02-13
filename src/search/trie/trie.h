#ifndef TRIE_TRIE_H
#define TRIE_TRIE_H

#include <vector>

#include "trie_node.h"
#include "trie_iterator.h"

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

private:
    tnode<T> *root;
    int size;
};

template <typename T>
trie<T>::trie(): size(0) {
    T flag = NULL;
    root = new tnode<T>(flag, nullptr, -1);
    size = 0;
}

template <typename T>
void trie<T>::insert(std::vector<int> key, T val) {
    tnode<T>* node = this->root;
    for (int& v : key) {
        if (node->getChild(v) != nullptr) {
            node = node->getChild(v);
        } else {
            T flag = NULL;
            tnode<T>* _node = new tnode<T>(flag, node, v);
            node->addChild(_node, v);
            node = node->getChild(v);
        }
    }
    if (!node->isEnd()) {
        this->size += 1;
    }
    node->update(val);
    node->markEnd(key);
}

template <typename T>
bool trie<T>::exist(std::vector<int> key) {
    bool res = true;
    tnode<T>* node = this->root;
    for (int& v : key) {
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
    return this->size == 0;
}

template <typename T>
typename trie<T>::iterator trie<T>::begin() {
    trie_iterator<T> it = *(new trie_iterator<T>(this->root));
    return ++it;
}

template <typename T>
tnode<T>* rbrecur(tnode<T>* n, int offset = 127, tnode<T>* r = nullptr);

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
        return rbrecur(it, 127, r);
    }
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
} // namespace trie

#endif
