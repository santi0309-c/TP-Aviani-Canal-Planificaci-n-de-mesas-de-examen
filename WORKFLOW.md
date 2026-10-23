# Flujo de trabajo Git — TP Aviani & Canal

## Estrategia de ramas

Usamos **ramas por ticket** + **Pull Requests** antes de mergear a `main`.

```
main          ← rama estable; solo recibe merges desde PRs
└── feat/T01-estructura-repo
└── feat/T02-dataset
└── feat/T03-modelo-predictivo
└── feat/T04-grafo-coloreo
└── feat/T05-api-backend
└── feat/T06-frontend
└── feat/T07-tests
└── feat/T08-informe
```

## Convención de nombres de ramas

```
feat/<id-ticket>-<descripcion-corta>
fix/<id-ticket>-<descripcion-corta>
```

## Convención de commits

```
<tipo>(<scope>): <descripción corta en presente>

Ejemplos:
feat(backend): agregar endpoint /cronograma
fix(graph): corregir coloreo cuando grafo es vacío
docs(readme): actualizar instrucciones de instalación
test(coloring): agregar test de validación de coloreo
```

## Flujo paso a paso

1. Crear rama desde `main`:
   ```bash
   git checkout main && git pull
   git checkout -b feat/T0X-nombre
   ```
2. Trabajar con commits chicos y descriptivos.
3. Abrir Pull Request hacia `main` cuando el ticket esté terminado.
4. El otro integrante revisa y aprueba.
5. Mergear con **Squash & Merge** o **Merge commit** (acordar uno).
6. Borrar la rama remota tras el merge.

## Reglas generales

- No pushear directo a `main`.
- Cada PR debe tener al menos **1 review** del otro integrante.
- Resolver conflictos en la rama feature, no en `main`.
- Mantener `main` siempre en estado funcional (tests pasando).
