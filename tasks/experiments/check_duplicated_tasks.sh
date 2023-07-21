# usage: ./check_duplicated_tasks.sh tasks_folder

for i in {0..49}; do
    for j in {0..49}; do
        if [ $i != $j ]; then
            cmp --silent "$1/p${i}.pddl" "$1/p${j}.pddl"
            if [ "$?" -eq "0" ]; then
                echo "DUPLICATED TASKS! $i and $j"
            fi
        fi
    done
done
