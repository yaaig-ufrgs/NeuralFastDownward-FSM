(define (problem blocks-14-0)
(:domain blocks)
(:objects i d b l c k m h j n e f g a )
		(:init
		(clear c)
		(clear d)
		(clear f)
		(clear k)
		(holding g)
		(on b a)
		(on c j)
		(on d i)
		(on f l)
		(on h m)
		(on i n)
		(on k b)
		(on l h)
		(on m e)
		(ontable a)
		(ontable e)
		(ontable j)
		(ontable n)
		)
(:goal (and (on e l) (on l f) (on f b) (on b j) (on j i) (on i n) (on n c)
            (on c k) (on k g) (on g d) (on d m) (on m a) (on a h)))
)
