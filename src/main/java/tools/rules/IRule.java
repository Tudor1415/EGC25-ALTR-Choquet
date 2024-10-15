/*
 * This file is part of io.gitlab.chaver:data-mining (https://gitlab.com/chaver/data-mining)
 *
 * Copyright (c) 2022, IMT Atlantique
 *
 * Licensed under the MIT license.
 *
 * See LICENSE file in the project root for full license information.
 */
package tools.rules;

import java.util.Set;

/**
 * Represents an association rule with antecedent x and consequent y, such that
 * z = x U y
 */
public interface IRule {

    Set<String> getItemsInX();

    String getY();

    int getFreqX();

    int getFreqY();

    int getFreqZ();

    void setX(Set<String> itemValues);

    void setY(String itemValue);

    void addToX(String itemValue);

    void removeFromX(String itemValue);

}
