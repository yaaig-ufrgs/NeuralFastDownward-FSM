(define (problem blocks-14-0)
(:domain blocks)
(:objects i d b l c k m h j n e f g a )
		(:init
		(clear c)
		(clear d)
		(clear k)
		(handempty)
		(on b e)
		(on c f)
		(on d i)
		(on f m)
		(on g a)
		(on h l)
		(on i n)
		(on j g)
		(on k h)
		(on l b)
		(on n j)
		(ontable a)
		(ontable e)
		(ontable m)
		)
(:goal (and (on e l) (on l f) (on f b) (on b j) (on j i) (on i n) (on n c)
            (on c k) (on k g) (on g d) (on d m) (on m a) (on a h)))
)
